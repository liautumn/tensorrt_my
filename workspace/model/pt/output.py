from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model

# Export the model
# ONNX ===> imgsz(h,w), half, dynamic, simplify, opset, batch
# TensorRT ===> imgsz, half, dynamic, simplify, workspace, int8, batch
model.export(format='onnx',
             imgsz=(1024, 1024),
             half=True,
             dynamic=False,
             simplify=True,
             batch=1
             )
