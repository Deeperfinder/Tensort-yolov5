// nvidia cuda
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>

// opencv
#include <opencv4/opencv2/opencv.hpp>

// common
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;


class yolov5{
	public:
		yolov5();
		void init();
		void buildEngine();
		void preProcess(const string& img_path);
		void infer();
		void postProcess();
		float letterbox(const cv::Mat& image, int stride, const cv::Scalar& color);
		cv::Mat convertHWCtoCHW(const cv::Mat& input);
	private:
		int m_output_area;
		int m_total_objects;
		int input_width;
		int input_height;

		// img 
		int img_area;
		std::vector<float> blob;
		cv::Mat m_resized_img;
		cv::Mat m_converted_img;
		cv::Mat m_normalized_img;
		cv::Mat m_process_img;

	protected:
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
};

class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) noexcept override{
		// suppress info-level messages
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
} gLogger;