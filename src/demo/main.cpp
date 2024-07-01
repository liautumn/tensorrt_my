#include <opencv2/opencv.hpp>
#include "infer.h"
#include "yolo.h"
#include "config.h"

using namespace std;

void syncInfer() {
    Config config;
    auto yolo = yolo::load(config.MODEL,
                           yolo::Type::V8);
    if (yolo == nullptr) return;

    cv::Mat mat = cv::imread(config.TEST_IMG);
    auto image = yolo::Image(mat.data, mat.cols, mat.rows);

    trt::Timer timer;

    while (true) {
        timer.start();
        auto objs = yolo->forward(image);
        timer.stop("batch one");
    }
}

int main() {
    syncInfer();
    return 0;
}
