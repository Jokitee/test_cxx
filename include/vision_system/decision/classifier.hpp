#ifndef VISION_SYSTEM_DECISION_CLASSIFIER_HPP
#define VISION_SYSTEM_DECISION_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "vision_system/core/types.hpp"
#include <string>
#include <vector>

// 新的 ONNX 分类模型算法
class FeatureDetectorONNX {
private:
    cv::dnn::Net net_;
    std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "sentry", "outpost", "base", "not_armor"};
    
public:
    FeatureDetectorONNX();
    std::string detect(const cv::Mat& inputImg, lightbors& armor_light);
};

#endif // VISION_SYSTEM_DECISION_CLASSIFIER_HPP
