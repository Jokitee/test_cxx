#ifndef VISION_SYSTEM_PROCESSING_IMAGE_PROCESSOR_HPP
#define VISION_SYSTEM_PROCESSING_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include "vision_system/core/types.hpp"
#include <vector>

// 角度矫正函数(针对于Opencv矩形拟合算法)
float getright_angle(const cv::RotatedRect& rect);

// 长边拉长函数
cv::RotatedRect stretchLongSide(const cv::RotatedRect& rect, float scale);

// 角点交换函数
void points_exchange(cv::Point2f points[4]);

// 装甲板计算函数
void set_light(lightbors& armor_light);

// 获取待检测的ROI
void CropAndResize(const lightbors& armor_light, cv::Mat& endB, cv::Mat& outImage);

#endif // VISION_SYSTEM_PROCESSING_IMAGE_PROCESSOR_HPP
