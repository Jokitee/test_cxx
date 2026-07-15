#ifndef VISION_SYSTEM_OUTPUT_VISUALIZER_HPP
#define VISION_SYSTEM_OUTPUT_VISUALIZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

void drawCube(cv::Mat& image,
              const cv::Mat& R_wo,
              const cv::Mat& t_wo,
              const cv::Mat& K,
              const cv::Mat& dist);

#endif // VISION_SYSTEM_OUTPUT_VISUALIZER_HPP
