/**
 * @brief 传统视觉装甲板(灯条)检测逻辑
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/decision/armor_detector.hpp"
#include "vision_system/processing/image_processor.hpp"

ArmorDetector::ArmorDetector(const DetectorConfig& cfg) : cfg_(cfg) {}

std::vector<lightbors> ArmorDetector::detect(const cv::Mat& matImage) {
    std::vector<lightbors> armor;
    if(matImage.empty()) return armor;

    std::vector<cv::Mat> channels;
    cv::split(matImage, channels);
    if(channels.size() < 3) return armor;
    
    cv::Mat r = channels[2];
    cv::Mat mask;
    cv::threshold(r, mask, cfg_.thresh_val, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> counters;
    cv::findContours(mask, counters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> end_rects;

    for (auto& cnt : counters) {
        cv::RotatedRect rotRect = cv::minAreaRect(cnt);
        float area = rotRect.size.width * rotRect.size.height;
        if (area < cfg_.area_min || area > cfg_.area_max) continue;

        float width = std::min(rotRect.size.width, rotRect.size.height);
        float height = std::max(rotRect.size.width, rotRect.size.height);
        float ratio = height / width;
        if (ratio < cfg_.ratio_min || ratio > cfg_.ratio_max) continue;

        float angle = getright_angle(rotRect);
        if (angle < cfg_.angle_min || angle > cfg_.angle_max) continue;

        end_rects.push_back(rotRect);
    }

    if (end_rects.size() >= 2) {
        for (size_t i = 0; i < end_rects.size() - 1; i++) {
            for (size_t j = i + 1; j < end_rects.size(); j++) {
                float angle_TF = getright_angle(end_rects[i]) - getright_angle(end_rects[j]);
                if (std::fabs(angle_TF) > 8.5) continue;
                
                float first_max = std::max(end_rects[i].size.width, end_rects[i].size.height);
                float second_max = std::max(end_rects[j].size.width, end_rects[j].size.height);
                float getheight = std::sqrt(std::pow(end_rects[i].center.x - end_rects[j].center.x, 2) + std::pow(end_rects[i].center.y - end_rects[j].center.y, 2));
                float getlight = (first_max + second_max) / 2;
                float distance_TF = getheight / getlight;
                
                if (distance_TF > cfg_.distance_max || distance_TF < cfg_.distance_min) continue;

                lightbors armor_light;
                if (end_rects[i].center.x < end_rects[j].center.x) {
                    armor_light.left_lightbors = end_rects[i];
                    armor_light.right_lightbors = end_rects[j];
                } else {
                    armor_light.left_lightbors = end_rects[j];
                    armor_light.right_lightbors = end_rects[i];
                }

                set_light(armor_light);
                armor.push_back(armor_light);
            }
        }
    }
    return armor;
}
