#include "yolov5.hpp"

cv::Mat yolov5::convertHWCtoCHW(const cv::Mat& input){
    CV_Assert(input.type() == CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    
    cv::Mat output = channels[0];
    return output;
}

yolov5::~yolov5()
{
    m_context.reset();
    m_engine.reset();
    // Release stream and buffers
    cudaStreamDestroy(m_stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));
}

void yolov5::init(const std::string onnx_path, std::string engine_path)
{
    input_height = 640;
    input_width = 640;
    bool save_engine = true;
    // 先检查是否有engine_path, 选择加载或者构建engine
    if(engine_path.empty())
    {
        std::cout << "[I] Can't find the engine file, start to build engine" <<std::endl;
        if(!buildEngine(onnx_path, save_engine)){
            std::cerr<<" Build engine false" << std::endl;
            exit(0);
        }
        else{
            engine_path = "/work/simple_yolov5_demo/model/yolov5_helmet.engine";
            std::cout << "[I] Build engine success" <<std::endl;
        };
        
        if(save_engine)
        {
            std::cout << "[I] Finding the engine file, start to load engine" <<std::endl;
            if(!loadEngine(engine_path)){
                std::cerr<<" load engine false" << std::endl;
            }
            else{
                std::cout << "[I] load engine success" <<std::endl;
            };
        }
    }
    else{
        // 直接加载engine
        if(!loadEngine(engine_path)){
            std::cerr<<" load engine false" << std::endl;
        }
        else{
            std::cout << "[I] load engine success" <<std::endl;
        };
    }


    
    //分配内存空间
    if(!AllocateGPUMemory()){
        std::cerr<<" Allocate cuda memory false!" << std::endl;
    };
}

float yolov5::letterbox(const cv::Mat& image, int stride=32)
{
    const cv::Scalar& color = cv::Scalar(114,114,114);
    cv::Size raw_shape = image.size();
    float ratio = std::min(
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
    
    letter_top = int(std::round(dh - 0.1f));
    letter_bottom = int(std::round(dh + 0.1f));
    letter_left = int(std::round(dw - 0.1f));
    letter_right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, m_resized_img, letter_top, letter_bottom, letter_left, letter_right, cv::BORDER_CONSTANT, color);
    return ratio;
    
}
yolov5::yolov5(const std::string onnx_path, const std::string engine_path, float initconfThreshold):m_confThreashold(initconfThreshold)
{
    init(onnx_path, engine_path);
}
void yolov5::preProcess(const string& img_path)
{

    auto image = cv::imread(img_path);
    // cv::resize(image, m_resized_img, cv::Size(input_width, input_height));
    m_ratio = letterbox(image);
    //cv::imwrite("/work/simple_yolov5_demo/test letterbox1.jpg", m_resized_img);

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

bool yolov5::loadEngine(const std::string engine_path)
{
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    // m_engine = std::unique_ptr<nvinfer1::ICudaEngine>
    // 读取序列化的引擎文件
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Unable to open engine file: "<< engine_path << std::endl;
        return false;
    }
    engineFile.seekg(0, engineFile.end);
    // 获取文件大小
    long int fsize = engineFile.tellg();
    // 将文件指针移回文件开头
    engineFile.seekg(0, engineFile.beg);

    // 分配足够的内存来容纳整个引擎文件
    std::unique_ptr<char[]> engineData(new char[fsize]);
    engineFile.read(engineData.get(), fsize);
    if (!engineFile)
    {
        std::cerr << "Unable to read engine file: "<< engine_path << std::endl;
        return false;
    }
    // 反序列引擎
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engineData.get(), fsize, nullptr));
    if (!m_engine)
    {
        std::cerr << "Unable to deserialize CUDA engine" << std::endl;
        return false;
    }
    // 创建执行上下文
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context)
    {
        std::cerr << "Unable to create execution context" << std::endl;
        return false;
    }
    return true;
}


bool yolov5::buildEngine(const std::string onnx_path,  bool save_engine)
{
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, gLogger);
    
    // 解析ONNX模型
    if(!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr<<"ERROR: could not parse the ONNX model." << std::endl;
        return false;
    }
    std::cout << std::string("[I] Succeeded parsing .onnx file!") << std::endl;
    // 构建推理引擎
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U<<20); 
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    
    // 将模型序列化到engine文件中
    if (save_engine)
    {
        nvinfer1::IHostMemory *serialized_model = m_engine->serialize();
        std::ofstream engineFile("/work/simple_yolov5_demo/model/yolov5_helmet.engine", std::ios::binary);
        if(!engineFile){
            std::cerr << "Cannot open engine file." << std::endl;
            return false;
        }
        engineFile.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
        engineFile.close();
        // 由于序列化引擎包含权重的必要拷贝，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除： 
        serialized_model->destroy();
        m_engine.reset();
        network->destroy();
        parser->destroy();
        config->destroy();
        builder->destroy();
        return true;
    }
    else{
        // 创建执行上下文
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context)
        {
            std::cerr << "Unable to create execution context" << std::endl;
            m_runtime->destroy();
            return false;
        }
        delete parser;
        delete network;
        delete config;
        delete builder;

        return true;
    }
}

bool yolov5::AllocateGPUMemory()
{
    // 检查engine指针是否有效
    if(!m_engine)
    {
        std::cerr << "Unable to find engine." << std::endl;
        return false;
    }
    // 获取模型输入尺寸并分配内存
    nvinfer1::Dims input_dim = m_engine->getBindingDimensions(0);
    int input_size = std::accumulate(input_dim.d, input_dim.d + input_dim.nbDims, 1 , std::multiplies<int>());

    // 分配输入和输出内存
    inputIndex = m_engine->getBindingIndex("images");
    outputIndex1 = m_engine->getBindingIndex("output");
    outputIndex2 = m_engine->getBindingIndex("414");
    outputIndex3 = m_engine->getBindingIndex("416");
    
    cudaError_t status = cudaMalloc(&buffers[inputIndex], input_size * sizeof(float));
    if (status != cudaSuccess) 
    {
        std::cerr << "cudaMalloc false." << std::endl;
        return false;  // 处理分配失败的情况
    }

    // 获取模型输出尺寸并分配GPU内存
    for(int i=1; i < m_engine->getNbBindings(); i++)
    {
        nvinfer1::Dims output_dim = m_engine->getBindingDimensions(i);
        int output_size = std::accumulate(output_dim.d, output_dim.d + output_dim.nbDims, 1, std::multiplies<int>());
        status = cudaMalloc(&buffers[i], output_size * sizeof(float));
        
        if (status != cudaSuccess) {
            buffers[0]=nullptr;  // 清理之前分配的内存
            return false;        // 处理分配失败的情况
        }
        m_outputData.push_back(std::vector<float>(output_size));
    }
    // 创建stream
    CHECK(cudaStreamCreate(&m_stream));

    return true;
}

void yolov5::infer()
{   
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], blob.data(), 3 * input_width * input_height * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    
    m_context->enqueueV2(buffers, m_stream, nullptr);
  
    CHECK(cudaMemcpyAsync(m_outputData[outputIndex1-1].data(), buffers[outputIndex1], m_outputData[outputIndex1-1].size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    CHECK(cudaMemcpyAsync(m_outputData[outputIndex2-1].data(), buffers[outputIndex2], m_outputData[outputIndex2-1].size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    CHECK(cudaMemcpyAsync(m_outputData[outputIndex3-1].data(), buffers[outputIndex3], m_outputData[outputIndex3-1].size() * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    
    cudaStreamSynchronize(m_stream);
}


float IOU(const Object& a, const Object& b){
    float areaA = a.rect.width * a.rect.height;
    if (areaA <= 0) return 0;

    float areaB = b.rect.width * b.rect.height;
    if (areaB <=0) return 0;

    float intersectionMinX = std::max(a.rect.x, b.rect.x);
    float intersectionMinY = std::max(a.rect.y, b.rect.y);
    float intersectionMaxX = std::min(a.rect.x+a.rect.width, b.rect.x+b.rect.width);
    float intersectionMaxY = std::min(a.rect.y+a.rect.height, b.rect.y+b.rect.height);

    float intersectionArea = std::max(intersectionMaxX - intersectionMinX, 0.0f) *
                             std::max(intersectionMaxY - intersectionMinY, 0.0f);
    return intersectionArea / (areaA + areaB - intersectionArea);
}


// NMS实现
std::vector<Object> NMS(const std::vector<Object>& boxes, float threshold=0.45){
    std::vector<Object> remainingBoxes = boxes;
    std::sort(remainingBoxes.begin(), remainingBoxes.end(), [](const Object& a, const Object& b)
    {
        return a.prob>b.prob;
    });
    
    std::vector<Object> pickedBoxes;

    while(!remainingBoxes.empty()){
        // 把得分最高的框框加入到结果列表中
        const Object currentBox = remainingBoxes.front();
        pickedBoxes.push_back(currentBox);

        // 移除最开始的框
        remainingBoxes.erase(remainingBoxes.begin());

        for(auto it = remainingBoxes.begin(); it != remainingBoxes.end(); ){
            if (IOU(currentBox, *it) > threshold){
                // 移除与当前框Iou大于阈值的所有框
                it = remainingBoxes.erase(it);
            }
            else{
                ++it;
            }
        }
    }
    return pickedBoxes;
}
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}
void yolov5::generate_proposal(std::vector<Object> & output_data)
{
    m_numclass = (m_outputData[2].size() / 1200) -5;
    int feature_area[3] = {6400, 1600, 400};
    const std::vector<std::vector<std::vector<int>>> anchors{
        {{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}
    };
    // per feature map
    // 解析80*80 40*40 20*20
    for(size_t i = 0; i<3; i++)
    {
        int feat_w = sqrt(feature_area[i]);
        for(int anchor_idx=0; anchor_idx<3; anchor_idx++)
        {
            for(int num=0; num<feature_area[i]; num++)
            {
                float score = m_outputData[i][feature_area[i] * ((5 + m_numclass) * anchor_idx + 4) + num];
                if (score>m_confThreashold)
                {
                    float t_x = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + 0)) + num];
                    float t_y = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + 1)) + num];
                    float t_w = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + 2)) + num];
                    float t_h = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + 3)) + num];
                    
                    int max_cls_index =0;
                    float max_value = 0;
                    for(int x=0; x<m_numclass; x++)
                    {
                        float cls_value = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + 5 + x)) + num];
                        if(cls_value > max_value){
                            max_value=cls_value;
                            max_cls_index = x;
                        }
                    }
                    Object obj;
                    obj.rect.x = (t_x * 2 - 0.5 + num % feat_w) / feat_w * input_width;
                    obj.rect.y = (t_y * 2 - 0.5 + num / feat_w) / feat_w * input_height;
                    obj.rect.width = pow(t_w * 2, 2) * anchors[i][anchor_idx][0];
                    obj.rect.height = pow(t_h * 2, 2) * anchors[i][anchor_idx][1];

                    obj.rect.x = obj.rect.x - float(obj.rect.width/2);
                    obj.rect.y = obj.rect.y - float(obj.rect.height/2);
                     
                    obj.class_id = 5 + max_cls_index;
                    obj.prob = m_outputData[i][(feature_area[i] * ((5 + m_numclass)*anchor_idx + obj.class_id)) + num] * score;
                    output_data.push_back(obj);
                }
            }
        }
    }

}

std::vector<Object> yolov5::postProcess()
{
    std::vector<Object> proposals;
    generate_proposal(proposals);
    // draw_box(proposals);
    // NMS
    std::vector<Object> nms_result = NMS(proposals);
    draw_box(nms_result);
    
    //rescale box
    for(auto box = nms_result.begin(); box!= nms_result.end(); box++){
        box->rect.x = (box->rect.x-letter_left) / m_ratio;
        box->rect.y = (box->rect.y-letter_top) / m_ratio;
        box->rect.height = box->rect.height / m_ratio;
        box->rect.width = box->rect.width / m_ratio;
    }

    return nms_result;
}

void draw_box(std::vector<Object> boxes, std::string img_path){
    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cout<< "false to read image!" << std::endl;
        exit(-1);
    }
    for(auto box = boxes.begin(); box!= boxes.end(); box++){
        cv::Point topleft(box->rect.x, box->rect.y);
        cv::Rect rect(
            topleft, 
            cv::Point(topleft.x + box->rect.width, topleft.y + box->rect.height)
        );
        cv::rectangle(image, rect, (0,0,255));
    }
    cv::imwrite("./result/nms_result.jpg", image);
}


void yolov5::draw_box(std::vector<Object> boxes){
    cv::Mat image = m_resized_img;
    if(image.empty()){
        std::cout<< "false to read image!" << std::endl;
        exit(-1);
    }
    for(auto box = boxes.begin(); box!= boxes.end(); box++){
        cv::Point topleft(box->rect.x, box->rect.y);
        cv::Rect rect(
            topleft, 
            cv::Point(topleft.x + box->rect.width, topleft.y + box->rect.height)
        );
        cv::rectangle(image, rect, (0,0,255));
    }
    cv::imwrite("./result/result.jpg", image);
}