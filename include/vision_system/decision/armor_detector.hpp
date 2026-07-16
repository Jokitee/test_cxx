#ifndef VISION_SYSTEM_DECISION_ARMOR_DETECTOR_HPP
#define VISION_SYSTEM_DECISION_ARMOR_DETECTOR_HPP

#include "vision_system/core/types.hpp"
#include "vision_system/core/config_manager.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class ArmorDetector {
public:
    ArmorDetector(const DetectorConfig& cfg);
    std::vector<lightbors> detect(const cv::Mat& matImage);
    const std::vector<cv::RotatedRect>& getLightbars() const { return lightbars_; }
private:
    DetectorConfig cfg_;
    std::vector<cv::RotatedRect> lightbars_;
};

#endif
