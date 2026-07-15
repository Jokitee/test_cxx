#ifndef VISION_SYSTEM_CORE_TYPES_HPP
#define VISION_SYSTEM_CORE_TYPES_HPP

#include <opencv2/opencv.hpp>
#include <string>

struct lightbors {
    cv::RotatedRect left_lightbors;
    cv::RotatedRect right_lightbors;
    cv::RotatedRect left_armor;
    cv::RotatedRect right_armor;
    cv::Point2f armor_center;
    cv::Point2f armor_point[4];     // 左上 -> 右上 -> 右下 -> 左下
    std::string ID;                 // 装甲板数字
    bool righting = false;          // 是否识别到有效数字

    std::string test_ID;            // 反馈当前检测各参数
};

#endif // VISION_SYSTEM_CORE_TYPES_HPP
