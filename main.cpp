#include <iostream>
#include "yolov5.cpp"
int main(int, char**){
    std::cout << "Hello, from YOLOV5!\n";
    yolov5 yolo;
    const std::string img_path = "/work/simple_yolov5_demo/hard_hat_workers101.png";
    yolo.preProcess(img_path);
}
