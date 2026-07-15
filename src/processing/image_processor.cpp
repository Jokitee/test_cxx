/**
 * @brief 图像处理与几何变换工具库
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/processing/image_processor.hpp"
#include <cmath>
#include <algorithm>

float getright_angle(const cv::RotatedRect& rect) {
    float angle = rect.angle;
    if (rect.size.width < rect.size.height)
        angle += 90.f;

    if (angle < 0.f)   angle += 180.f;
    if (angle >= 180.f) angle -= 180.f;
    return angle;
}

cv::RotatedRect stretchLongSide(const cv::RotatedRect& rect, float scale) {
    cv::RotatedRect result = rect;
    if (rect.size.width >= rect.size.height) {
        result.size.width *= scale;
    } else {
        result.size.height *= scale;
    }
    return result;
}

void points_exchange(cv::Point2f points[4]) {
    std::vector<cv::Point2f> pts(points, points + 4);
    
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (std::abs(a.y - b.y) > 1e-5) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    
    cv::Point2f top1 = pts[0];
    cv::Point2f top2 = pts[1];
    cv::Point2f bottom1 = pts[2];
    cv::Point2f bottom2 = pts[3];
    
    cv::Point2f top_left, top_right;
    if (top1.x < top2.x) {
        top_left  = top1;
        top_right = top2;
    } else {
        top_left  = top2;
        top_right = top1;
    }
    
    cv::Point2f bottom_left, bottom_right;
    if (bottom1.x < bottom2.x) {
        bottom_left  = bottom1;
        bottom_right = bottom2;
    } else {
        bottom_left  = bottom2;
        bottom_right = bottom1;
    }
    
    points[0] = top_left;
    points[1] = top_right;
    points[2] = bottom_right;
    points[3] = bottom_left;
}

void set_light(lightbors& armor_light){
    armor_light.left_lightbors = stretchLongSide(armor_light.left_lightbors, 2.4f);
    armor_light.right_lightbors = stretchLongSide(armor_light.right_lightbors, 2.4f);
    cv::Point2f left_points[4];
    cv::Point2f right_points[4];
    armor_light.left_lightbors.points(left_points);
    armor_light.right_lightbors.points(right_points);
    points_exchange(left_points);
    points_exchange(right_points);

    for(size_t i = 0 ; i < 4; i++){
        if (i == 0 || i == 3) {
            armor_light.armor_point[i] = left_points[i];
        } else {
            armor_light.armor_point[i] = right_points[i];
        }
    }
    armor_light.armor_center = {(armor_light.left_armor.center.x + armor_light.right_armor.center.x) / 2 , (armor_light.left_armor.center.y + armor_light.right_armor.center.y) / 2};
}

void CropAndResize(const lightbors& armor_light, cv::Mat& endB, cv::Mat& outImage) {
    float min_x = std::min({armor_light.armor_point[0].x, armor_light.armor_point[1].x, armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float max_x = std::max({armor_light.armor_point[0].x, armor_light.armor_point[1].x, armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float min_y = std::min({armor_light.armor_point[0].y, armor_light.armor_point[1].y, armor_light.armor_point[2].y, armor_light.armor_point[3].y});
    float max_y = std::max({armor_light.armor_point[0].y, armor_light.armor_point[1].y, armor_light.armor_point[2].y, armor_light.armor_point[3].y});

    int roi_left = std::max(static_cast<int>(min_x), 0);
    int roi_top = std::max(static_cast<int>(min_y), 0);
    int roi_right = std::min(static_cast<int>(max_x), endB.cols);
    int roi_bottom = std::min(static_cast<int>(max_y), endB.rows);

    cv::Rect roi(roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top);

    if (roi.area() > 0) {
        outImage = endB(roi).clone();
    } else {
        outImage = cv::Mat();
    }
}
