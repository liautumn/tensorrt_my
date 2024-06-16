cmake_minimum_required(VERSION 3.28)
project(tensorrt_my CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 定义头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(OpenCV_DIR D:/cuda/opencv/build)
find_package(OpenCV REQUIRED)
set(TensorRT_DIR D:/cuda/TensorRT-8.6.1.6)
set(CUDA_DIR C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1)

# 设置包含路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_DIR}/include
        ${TensorRT_DIR}/samples/common
        ${CUDA_DIR}/include
)

# 设置库路径
link_directories(
        ${TensorRT_DIR}/lib
        ${CUDA_DIR}/lib/x64
)

# 定义链接的库
set(LIBS
        ${OpenCV_LIBS}
        nvinfer
        nvinfer_plugin
        nvonnxparser
        cuda
        cublas
        cudart
        cudnn
)

add_executable(${PROJECT_NAME}
        include/yolo.h
        include/infer.h
        src/infer.cu
        src/yolo.cu
        src/demo/config.h
        src/demo/main.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# 链接库
target_link_libraries(${PROJECT_NAME} ${LIBS})
