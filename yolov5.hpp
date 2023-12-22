#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>

using namespace std;

class yolov5{
	public:
	
		void init();
		void buildEngine();
		void preProcess(const string& img_path);
		void infer();
		void postProcess();
		cv::Mat convertHWCtoCHW(const cv::Mat& input);
	private:
		int m_output_area;
		int m_total_objects;
		int input_width;
		int input_height;
		

		// img 
		int img_area;
		cv::Mat m_resized_img;
		cv::Mat m_converted_img;
		cv::Mat m_normalized_img;
		cv::Mat m_process_img;
};