#include <iostream>
#include "yolov5.cpp"
int main(int, char**){
    std::cout << "Hello, from YOLOV5!\n";

    const std::string onnx_path = "/work/simple_yolov5_demo/gddi_model.onnx";
    std::string engine_path = "/work/simple_yolov5_demo/yolov5_helmet.engine";
    float confth = 0.5;

    yolov5 yolo(onnx_path, engine_path, confth);
    const std::string img_path = "/work/simple_yolov5_demo/images/hard_hat_workers101.png";
    yolo.preProcess(img_path);
    yolo.infer();
    yolo.postProcess();
}
