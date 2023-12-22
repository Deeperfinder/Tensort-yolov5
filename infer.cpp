#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace cv;
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

int main()
{

    std::cout<<"hallo"<<std::endl;
    const int batchSize = 1;
    const int inputH = 224;
    const int inputW = 224;
    const int inputC = 3;

    cv::Mat img = cv::imread("../person.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image." << std::endl;
        return -1;
    }
    // Resize the image to the input size required by the model
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(inputW, inputH));

    // Convert BGR to RGB (OpenCV uses BGR by default)
    cv::Mat rgbImg;
    cv::cvtColor(resizedImg, rgbImg, cv::COLOR_BGR2RGB);

    // Normalize the image if required by the model
    rgbImg.convertTo(rgbImg, CV_32F, 1.f / 255.f);



    // 创建TensorRT模型和推理引擎
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto config = builder -> createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, gLogger);

    // 解析ONNX模型
    if(!parser->parseFromFile("../resnet50.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr<<"ERROR: could not parse the ONNX model." << std::endl;
        return -1;
    }
    // 构建推理引擎
    config->setMaxWorkspaceSize(1<<20);
    auto engine = builder->buildEngineWithConfig(*network, *config);
    
    // 释放中间资源
    parser->destroy();
    network->destroy();
    config->destroy();

    // 创建推理上下文
    auto context = engine->createExecutionContext();

    // 分配输入和输出内存
    void* buffers[2]; // one input and one output
    const int inputIndex = engine->getBindingIndex("input.1");
    const int outputIndex = engine->getBindingIndex("495");

    // 在GPU上分配内存
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * 224 * 224 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * 1000 * sizeof(float)); // Assuming 1000 output classes

    // 创建一个随机输入数据
    //std::vector<float> inputData(batchSize * 3 * 224 * 224);
    std::vector<float> inputData(batchSize * inputH * inputW * inputC);
    cudaMemcpy(buffers[inputIndex], inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 执行推理
    context -> executeV2(buffers);

    std::vector<float> outputData(batchSize * 1000);
    // 复制结果到cpu
    cudaMemcpy(outputData.data(), buffers[outputIndex], outputData.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    builder->destroy();
    std::cout<<outputData.size() << std::endl;
    auto maxIt = std::max_element(outputData.begin(), outputData.end());
    size_t maxIndex = std::distance(outputData.begin(), maxIt);
    std::cout << "max:" << *maxIt << std::endl;
    std::cout << "max index:" << maxIndex<<std::endl;
    std::cout<<outputData[0]<<std::endl;
    return 0;
    
}