#include <string>
#include "infer.h"
#include "config.h"
#include <opencv2/opencv.hpp>

using namespace std;

enum class NormType : int {
    None = 0, MeanStd = 1, AlphaBeta = 2
};
enum class ChannelType : int {
    None = 0, SwapRB = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm {
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type) {
    Norm out;
    out.type = NormType::MeanStd;
    out.alpha = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std, std, sizeof(out.std));
    return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    out.channel_type = channel_type;
    return out;
}

Norm Norm::None() { return Norm(); }

shared_ptr<trt::Infer> trt_;
float confidence_threshold_, nms_threshold_;
vector<int> bbox_head_dims_;
int network_input_width_, network_input_height_, num_classes_ = 0;
bool isdynamic_model_ = false;
Norm normalize_;

bool load(const string &engine_file, float confidence_threshold, float nms_threshold) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;

    trt_->print();

    confidence_threshold_ = confidence_threshold;
    nms_threshold_ = nms_threshold;

    auto input_dim = trt_->static_dims(0);
    bbox_head_dims_ = trt_->static_dims(1);
    network_input_width_ = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_ = trt_->has_dynamic_dim();

    normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
    num_classes_ = bbox_head_dims_[2] - 4;
    return true;
}

struct Image {
    const void *bgrptr = nullptr;
    int width = 0, height = 0;

    Image() = default;

    Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {
    }
};

trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
// left, top, right, bottom, confidence, class,
const int NUM_BOX_ELEMENT = 8;
// keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;
vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;

void adjust_memory(int batch_size) {
    // the inference batch_size
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
    output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
    if ((int) preprocess_buffers_.size() < batch_size) {
        for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
            preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }
}

struct AffineMatrix {
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
        float scale_x = get<0>(to) / (float) get<0>(from);
        float scale_y = get<1>(to) / (float) get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
        uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
        int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;

    float m_x1 = warp_affine_matrix_2_3[0];
    float m_y1 = warp_affine_matrix_2_3[1];
    float m_z1 = warp_affine_matrix_2_3[2];
    float m_x2 = warp_affine_matrix_2_3[3];
    float m_y2 = warp_affine_matrix_2_3[4];
    float m_z2 = warp_affine_matrix_2_3[5];

    float src_x = m_x1 * dx + m_y1 * dy + m_z1;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_value;
        uint8_t *v2 = const_value;
        uint8_t *v3 = const_value;
        uint8_t *v4 = const_value;
        if (y_low >= 0) {
            if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
        }

        // same to opencv
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    if (norm.channel_type == ChannelType::SwapRB) {
        float t = c2;
        c2 = c0;
        c0 = t;
    }

    if (norm.type == NormType::MeanStd) {
        c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
        c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
        c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    } else if (norm.type == NormType::AlphaBeta) {
        c0 = c0 * norm.alpha + norm.beta;
        c1 = c1 * norm.alpha + norm.beta;
        c2 = c2 * norm.alpha + norm.beta;
    }

    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

static void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                     cudaStream_t stream) {
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
            src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
            matrix_2_3, norm);
}

void preprocess(int ibatch, const Image &image,
                shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                AffineMatrix &affine,
                void *stream = nullptr) {
    affine.compute(make_tuple(image.width, image.height),
                   make_tuple(network_input_width_, network_input_height_));

    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    size_t size_image = image.width * image.height * 3;
    size_t size_matrix = upbound(sizeof(affine.d2i), 32);
    uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
    float *affine_matrix_device = (float *) gpu_workspace;
    uint8_t *image_device = gpu_workspace + size_matrix;

    uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
    float *affine_matrix_host = (float *) cpu_workspace;
    uint8_t *image_host = cpu_workspace + size_matrix;

    // speed up
    cudaStream_t stream_ = (cudaStream_t) stream;
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                    cudaMemcpyHostToDevice, stream_);

    warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                             image.height, input_device, network_input_width_,
                                             network_input_height_, affine_matrix_device, 114,
                                             normalize_, stream_);
}


int GPU_BLOCK_THREADS = 1024;

static dim3 grid_dims(int numJobs) {
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float) numBlockThreads));
}

static dim3 block_dims(int numJobs) {
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v8(
        float *predict, // 边界框预测数组
        int num_bboxes, // 边界框数量
        int num_classes, // 类别数量
        int output_cdim, // 输出维度
        float confidence_threshold, // 置信度阈值
        float *invert_affine_matrix, // 逆仿射矩阵
        float *parray, // 输出数组，用于存储有效的边界框
        int MAX_IMAGE_BOXES // 最大边界框数量
) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int) *bboxes, MAX_IMAGE_BOXES);
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
//        if (i == position || pcurrent[5] != pitem[5]) continue;
        if (i == position) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static void decode_kernel_invoker(
        float *predict,
        int num_bboxes,
        int num_classes,
        int output_cdim,
        float confidence_threshold,
        float nms_threshold,
        float *invert_affine_matrix,
        float *parray,
        int MAX_IMAGE_BOXES,
        cudaStream_t stream
) {
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    decode_kernel_v8<<<grid, block, 0, stream>>>(
            predict,
            num_bboxes,
            num_classes,
            output_cdim,
            confidence_threshold,
            invert_affine_matrix,
            parray,
            MAX_IMAGE_BOXES
    );

    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold);
}

struct Box {
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left),
              top(top),
              right(right),
              bottom(bottom),
              confidence(confidence),
              class_label(class_label) {
    }
};

typedef vector<Box> BoxArray;

vector<BoxArray> forwards(const vector<Image> &images, void *stream = nullptr) {
    int num_image = images.size();
    if (num_image == 0) return {};

    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image) {
        if (isdynamic_model_) {
            infer_batch_size = num_image;
            input_dims[0] = num_image;
            if (!trt_->set_run_dims(0, input_dims)) return {};
        } else {
            if (infer_batch_size < num_image) {
                INFO(
                        "When using static shape model, number of images[%d] must be "
                        "less than or equal to the maximum batch[%d].",
                        num_image, infer_batch_size);
                return {};
            }
        }
    }
    adjust_memory(infer_batch_size);

    vector<AffineMatrix> affine_matrixs(num_image);
    cudaStream_t stream_ = (cudaStream_t) stream;
    for (int i = 0; i < num_image; ++i) {
        preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);
    }

    float *bbox_output_device = bbox_predict_.gpu();
    vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};
    if (!trt_->forward(bindings, stream)) {
        INFO("Failed to tensorRT forward.");
        return {};
    }

    for (int ib = 0; ib < num_image; ++ib) {
        float *boxarray_device =
                output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
        float *affine_matrix_device = (float *) preprocess_buffers_[ib]->gpu();
        float *image_based_bbox_output =
                bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
        cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_);
        decode_kernel_invoker(
                image_based_bbox_output,
                bbox_head_dims_[1],
                num_classes_,
                bbox_head_dims_[2],
                confidence_threshold_,
                nms_threshold_,
                affine_matrix_device,
                boxarray_device,
                MAX_IMAGE_BOXES,
                stream_
        );
    }
    cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    vector<BoxArray> arrout(num_image);
    for (int ib = 0; ib < num_image; ++ib) {
        float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
        int count = min(MAX_IMAGE_BOXES, (int) *parray);
        BoxArray &output = arrout[ib];
        output.reserve(count);
        for (int i = 0; i < count; ++i) {
            float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
            int label = pbox[5];
            int keepflag = pbox[6];
            if (keepflag == 1) {
                Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                output.emplace_back(result_object_box);
            }
        }
    }
    return arrout;
}

int main() {
    Config config;

    auto model = load(config.MODEL, 0.1, 0.4);
    if (!model) return 0;

    cv::Mat mat = cv::imread(config.TEST_IMG);
    vector<Image> images;
    for (int i = 0; i < 10; i++) {
        images.emplace_back(mat.data, mat.cols, mat.rows);
    }

    auto input_dims = trt_->static_dims(0);
    int infer_batch_size = input_dims[0];
    std::string message = "batch " + std::to_string(infer_batch_size);
    const char *c_message = message.c_str();

    trt::Timer timer;

    while (true) {
        timer.start();
        auto objs = forwards(images);
        timer.stop(c_message);
//
//    for (auto &obj: objs[0]) {
//        cout << "class_label: " << obj.class_label << " caption: " << obj.confidence << " (L T R B): (" << obj.left
//             << ", "
//             << obj.top << ", " << obj.right << ", " << obj.bottom << ")" << endl;
//
//        cv::rectangle(mat, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
//                      cv::Scalar(0, 255, 255), 5);
//
//        auto caption = cv::format("%d %.2f", obj.class_label, obj.confidence);
//        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
//        cv::rectangle(mat, cv::Point(obj.left - 3, obj.top - 33),
//                      cv::Point(obj.left + width, obj.top), cv::Scalar(0, 255, 255), -1);
//        cv::putText(mat, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
//                    16);
//
//    }
//    cv::imshow("yolov8", mat);  // 显示帧
//    cv::waitKey(0);
    }

    return 0;
}