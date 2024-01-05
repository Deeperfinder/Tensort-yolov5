// nvidia cuda
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>


// opencv
#include <opencv4/opencv2/opencv.hpp>

// common
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <numeric>    // std::accumulate
#include <functional> // std::multiplies
#include <fstream>

using namespace std;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

struct Object{
	cv::Rect_<float> rect;
	int class_id;
	float prob;
};

class yolov5{
	public:
		yolov5(const std::string onnx_path, const std::string engine_path, float initconfThreshold);
		// yolov5(std::shared_ptr<nvinfer1::ICudaEngine> engine):m_engine(engine);    // todo 多个对象复用engine
		~yolov5();
		
		void init(const std::string onnx_path, std::string engine_path);
		bool buildEngine(const std::string onnx_path, bool save_engine = true);
		bool loadEngine(const std::string engine_path);
		bool AllocateGPUMemory();


		void preProcess(const string& img_path);
		void infer();
		void postProcess();
		float letterbox(const cv::Mat& image, int stride, const cv::Scalar& color);
		void generate_proposal(std::vector<Object> & output_data);
		// std::shared_ptr<nvinfer1::ICudaEngine> get_shared_engine();   // todo 多个对象复用engine

		cv::Mat convertHWCtoCHW(const cv::Mat& input);
	private:
		int m_total_objects;
		int input_width;
		int input_height;

		//model
		float m_confThreashold;
		float m_numclass;
		// img 
		int img_area;
		std::vector<float> blob;
		float m_ratio;
		cv::Mat m_resized_img;
		cv::Mat m_converted_img;
		cv::Mat m_normalized_img;
		cv::Mat m_process_img;
				
		// buffers
		void* buffers[4];

		// nvidia
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
		std::unique_ptr<nvinfer1::IRuntime> m_runtime;
		std::unique_ptr<nvinfer1::IExecutionContext> m_context;
		cudaStream_t m_stream;

		int volatile inputIndex;
		int volatile outputIndex1;
		int volatile outputIndex2;
		int volatile outputIndex3;

		// output data
		std::vector<std::vector<float>> m_outputData;
};

class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) noexcept override{
		// suppress info-level messages
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
} gLogger;