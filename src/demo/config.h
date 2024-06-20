#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\yolov8s.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\car.jpg)";

    const char *cocolabels[80] = {
            "person", "bicycle", "car",
            "motorcycle", "airplane", "bus",
            "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird",
            "cat", "dog", "horse",
            "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup",
            "fork", "knife", "spoon",
            "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza",
            "donut", "cake", "chair",
            "couch", "potted plant", "bed",
            "dining table", "toilet", "tv",
            "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink",
            "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
};

#endif //YOLO_CONFIG_H
