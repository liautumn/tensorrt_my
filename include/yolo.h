#ifndef __YOLO_H__
#define __YOLO_H__

#include <future>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace yolo {
    enum class Type : int {
        V5 = 0,
        X = 1,
        V3 = 2,
        V7 = 3,
        V8 = 5
    };

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


    struct Image {
        const void *bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;

        Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {
        }
    };

    typedef vector<Box> BoxArray;

    // [Preprocess]: 0.50736 ms
    // [Forward]: 3.96410 ms
    // [BoxDecode]: 0.12016 ms
    // [SegmentDecode]: 0.15610 ms
    class Infer {
    public:
        virtual BoxArray forward(const Image &image, void *stream = nullptr) = 0;

        virtual vector<BoxArray> forwards(const vector<Image> &images,
                                          void *stream = nullptr) = 0;
    };

    shared_ptr<Infer> load(const string &engine_file, Type type,
                           float confidence_threshold = 0.45f, float nms_threshold = 0.5f);

    tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);

    tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
}; // namespace yolo

#endif  // __YOLO_H__
