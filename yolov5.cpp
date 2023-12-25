#include "yolov5.hpp"

cv::Mat yolov5::convertHWCtoCHW(const cv::Mat& input){
    CV_Assert(input.type() == CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    
    cv::Mat output = channels[0];
    return output;
}

void yolov5::init()
{
    input_height = 640;
    input_width = 640;
}

float yolov5::letterbox(const cv::Mat& image, int stride=32, const cv::Scalar& color = cv::Scalar(114,114,114))
{
    cv::Size raw_shape = image.size();
    volatile float ratio = std::min(
        (float)input_height / (float)raw_shape.height, (float)input_width / (float)raw_shape.width);
    int newUnpad[2]{
        (int)std::round((float)raw_shape.width * ratio), (int)std::round((float)raw_shape.height * ratio)
    };
    cv::Mat tmp;
    // 当shape为640的时候，直接拷贝过去就行，考虑 (640, 720) (640, 480)两种情况去理解
    if(raw_shape.width != newUnpad[0] || raw_shape.height != 640)
    {
        cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else{
        tmp = image.clone();
    }

    float dw = input_width - newUnpad[0];
    float dh = input_height - newUnpad[1];

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, m_resized_img, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return ratio;
    
}
yolov5::yolov5()
{
    init();
}
void yolov5::preProcess(const string& img_path)
{
    auto image = cv::imread(img_path);
    // cv::resize(image, m_resized_img, cv::Size(input_width, input_height));
    float ratio = letterbox(image);
    cv::imwrite("/work/simple_yolov5_demo/test letterbox1.jpg", m_resized_img);

    img_area = m_resized_img.cols * m_resized_img.rows;

    cv::cvtColor(m_resized_img, m_converted_img, cv::COLOR_BGR2RGB);
    cv::normalize(m_converted_img, m_normalized_img, cv::NORM_L2, 0, 1, CV_32FC3);

    // 将图像从HWC布局转为CHW布局
    blob.resize(m_normalized_img.total() * m_normalized_img.channels());
    for(int c=0; c<m_normalized_img.channels(); ++c){
        for(int i=0; i<m_normalized_img.rows; ++i){
            for(int j=0; j<m_normalized_img.cols; ++j){
                blob[c * m_normalized_img.rows * m_normalized_img.cols + i * m_normalized_img.cols + j ]=
                m_normalized_img.at<cv::Vec3f>(i,j)[c];
            }
        }
    }
}

void yolov5::buildEngine()
{
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, gLogger);
    
    // 解析ONNX模型
    if(!parser->parseFromFile("../gddi_model.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr<<"ERROR: could not parse the ONNX model." << std::endl;
       
    }
    
    // 构建推理引擎
    config->setMaxWorkspaceSize(1<<20);
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    
    // 释放中间资源
    parser->destroy();
    network->destroy();
    config->destroy();
}

void yolov5::infer()
{
    // 分配内存并将数据复制到GPU
    void *device_input_buffer_raw;
    cudaMalloc(&device_input_buffer_raw, blob.size() * sizeof(float));
    cudaMemcpy(device_input_buffer_raw, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    
    // 分配输入和输出内存
    void * buffers[4]; // 1 input and 3 output
    const int inputIndex = 
}