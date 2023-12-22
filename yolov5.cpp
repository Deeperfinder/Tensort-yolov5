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

yolov5::yolov5()
{
    init();
}
void yolov5::preProcess(const string& img_path)
{
    auto image = cv::imread(img_path);
    cv::resize(image, m_resized_img, cv::Size(input_width, input_height));
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