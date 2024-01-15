#include <iostream>
#include "yolov5.cpp"
int main(int, char**){
    std::cout << "[I] Hello, from YOLOV5!\n";

    const std::string onnx_path = "/work/simple_yolov5_demo/model/gddi_model.onnx";
    std::string engine_path = "";   //"/work/simple_yolov5_demo/yolov5_helmet.engine";
    float confth = 0.5;

    yolov5 yolo(onnx_path, engine_path, confth);
    const std::string img_path = "/work/simple_yolov5_demo/helmet2.jpeg";
    std::ifstream img_file(img_path);
    if(!img_file.good()){
        std::cout << "确保图片是否存在！" << std::endl;
    }
    std::cout<< "[I] begin to preProcess!" << std::endl;
    yolo.preProcess(img_path);
    std::cout<< "[I] begin to infer!" << std::endl;
    yolo.infer();

    std::cout<< "[I] begin to postProcess!" << std::endl;
    std::vector<Object> res_box = yolo.postProcess();

    std::cout<< "[I] begin to draw box!" << std::endl;
    draw_box(res_box, img_path);


}
