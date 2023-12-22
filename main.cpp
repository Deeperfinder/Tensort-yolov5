#include <iostream>
#include "yolov5.cpp"
int main(int, char**){
    std::cout << "Hello, from YOLOV5!\n";
    yolov5 yolo;
    const std::string img_path = "./a.jpg";
    yolo.preProcess(img_path);
}
