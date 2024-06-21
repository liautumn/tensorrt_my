#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

#include "string"

using namespace std;

class Config {
public:
    const string MODEL = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\model\engine\yolov8s.engine)";
    const string TEST_IMG = R"(D:\autumn\Documents\JetBrainsProjects\CLionProjects\tensorrt\workspace\images\car.jpg)";
};

#endif //YOLO_CONFIG_H
