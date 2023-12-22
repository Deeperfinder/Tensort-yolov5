#include "yolov5.hpp"

cv::Mat yolov5::convertHWCtoCHW(const cv::Mat& input){
    CV_Assert(input.type() == CV_32FC3);
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    
    cv::Mat output = channels[0];
    return output;
}


void yolov5::preProcess(const string& img_path)
{
    auto image = cv::imread(img_path);
    cv::resize(image, m_resized_img, cv::Size(input_width, input_height));
    img_area = m_resized_img.cols * m_resized_img.rows;

    cv::cvtColor(m_resized_img, m_converted_img, cv::COLOR_BGR2RGB);
    cv::normalize(m_converted_img, m_normalized_img, 0, 1, CV_32F);

}