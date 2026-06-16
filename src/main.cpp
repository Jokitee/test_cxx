#include <iostream>
#include "CameraApi.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <map>
#include <string>

using namespace cv;

unsigned char           * g_pRgbBuffer;

struct lightbors
{
    cv::RotatedRect left_lightbors;
    cv::RotatedRect right_lightbors;
    cv::RotatedRect left_armor;
    cv::RotatedRect right_armor;
    cv::Point2f armor_center;
    cv::Point2f armor_point[4];     // 次序为 左上 -> 右上 -> 右下 -> 左下
    std::string ID;         // 装甲板数字
    bool righting = 0;
};


// 长边拉长函数
cv::RotatedRect stretchLongSide(const cv::RotatedRect& rect, float scale) {
    cv::RotatedRect result = rect;
    
    // 比较 width 和 height，长边是较大的那个
    if (rect.size.width >= rect.size.height) {
        result.size.width *= scale;  // width 是长边，拉长它
    } else {
        result.size.height *= scale; // height 是长边，拉长它
    }
    
    return result;
}

// 角点交换函数
void points_exchange(Point2f points[4]) {
    std::vector<Point2f> pts(points, points + 4);
    
    // Step 1: 按 y 坐标升序排序（y 小在上，y 大在下）
    // y 相同则按 x 排序（x 小在左）
    std::sort(pts.begin(), pts.end(), [](const Point2f& a, const Point2f& b) {
        if (std::abs(a.y - b.y) > 1e-5) {
            return a.y < b.y;  // y 小的排在前面（上方）
        }
        return a.x < b.x;      // y 相同，x 小的排在前面
    });
    
    // Step 2: 分组 —— 上方两点（y 较小）和下方两点（y 较大）
    Point2f top1 = pts[0];   // y 最小
    Point2f top2 = pts[1];   // y 次小
    Point2f bottom1 = pts[2]; // y 次大
    Point2f bottom2 = pts[3]; // y 最大
    
    // Step 3: 上方两点按 x 排序，确定左上和右上
    Point2f top_left, top_right;
    if (top1.x < top2.x) {
        top_left  = top1;   // x 小 → 左上
        top_right = top2;   // x 大 → 右上
    } else {
        top_left  = top2;
        top_right = top1;
    }
    
    // Step 4: 下方两点按 x 排序，确定左下和右下
    Point2f bottom_left, bottom_right;
    if (bottom1.x < bottom2.x) {
        bottom_left  = bottom1;  // x 小 → 左下
        bottom_right = bottom2;  // x 大 → 右下
    } else {
        bottom_left  = bottom2;
        bottom_right = bottom1;
    }
    
    // Step 5: 按顺时针重新赋值：左上 → 右上 → 右下 → 左下
    points[0] = top_left;      // 左上
    points[1] = top_right;     // 右上
    points[2] = bottom_right;  // 右下
    points[3] = bottom_left;   // 左下
}

// 装甲板计算函数
void set_light(lightbors& armor_light){

    armor_light.left_lightbors = stretchLongSide(armor_light.left_lightbors, 2.2f);
    armor_light.right_lightbors = stretchLongSide(armor_light.right_lightbors, 2.2f);
    // 获取角点并放进数组内部
    Point2f left_points[4];
    Point2f right_points[4];
    armor_light.left_lightbors.points(left_points);
    armor_light.right_lightbors.points(right_points);
    points_exchange(left_points);
    points_exchange(right_points);

    for(size_t i = 0 ; i < 4; i++){
        if (i == 0 || i == 3)
        {
            armor_light.armor_point[i] = left_points[i];
        }else{
            armor_light.armor_point[i] = right_points[i];
        }
        
    }
    armor_light.armor_center = {(armor_light.left_armor.center.x + armor_light.right_armor.center.x) / 2 , (armor_light.left_armor.center.y + armor_light.right_armor.center.y) / 2};

}

/*
// // 透视变换获取图像 并进行特征变换变为 64 * 64
// void Per_transformation(const lightbors& armor_light, cv::Mat& endB, cv::Mat outImage){
    
//     int dst_w = 64;
//     int dst_h = 64;

//     std::vector<cv::Point2f> dst_pts = {
//         {0.0f,          0.0f},
//         {(float)dst_w,  0.0f},
//         {(float)dst_w,  (float)dst_h},
//         {0.0f,          (float)dst_h}
//     };

//     cv::Mat M = cv::getPerspectiveTransform(armor_light.armor_point, dst_pts.data());
//     std::cout << "[DEBUG] M empty: " << M.empty() << std::endl;

//     cv::warpPerspective(endB, outImage, M, cv::Size(dst_w, dst_h));

// }
*/

void CropAndResize(const lightbors& armor_light, cv::Mat& endB, cv::Mat& outImage) {
    
    int dst_w = 64;
    int dst_h = 64;

    // 计算 armor_point 四个角点的外接矩形
    float min_x = std::min({armor_light.armor_point[0].x, armor_light.armor_point[1].x, 
                            armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float max_x = std::max({armor_light.armor_point[0].x, armor_light.armor_point[1].x, 
                            armor_light.armor_point[2].x, armor_light.armor_point[3].x});
    float min_y = std::min({armor_light.armor_point[0].y, armor_light.armor_point[1].y, 
                            armor_light.armor_point[2].y, armor_light.armor_point[3].y});
    float max_y = std::max({armor_light.armor_point[0].y, armor_light.armor_point[1].y, 
                            armor_light.armor_point[2].y, armor_light.armor_point[3].y});

    // 限制在图像范围内
    int x = std::max(0, static_cast<int>(min_x));
    int y = std::max(0, static_cast<int>(min_y));
    int width  = std::min(static_cast<int>(max_x) - x, endB.cols - x);
    int height = std::min(static_cast<int>(max_y) - y, endB.rows - y);

    if (width <= 0 || height <= 0) {
        std::cerr << "[ERROR] Invalid crop region!" << std::endl;
        outImage = cv::Mat::zeros(dst_h, dst_w, endB.type());
        return;
    }

    // 截取 ROI 并缩放到 64x64
    cv::Rect roi(x, y, width, height);
    cv::Mat cropped = endB(roi).clone();
    cv::resize(cropped, outImage, cv::Size(dst_w, dst_h));
}

// 自研分类算法
void distinguish(lightbors& armor_light, std::vector<std::vector<cv::Point>>& contours){
    // 空集反馈
    if (contours.empty()){
        armor_light.ID = "UNKNOWN";
        armor_light.righting = 0;
        std::cout << "error" << std::endl;
        return;
    } 

    int max_idx = 0;
    double max_area = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if(area > max_area){
            max_area = area;
            max_idx = i;
        }
    }
    auto& contour = contours[max_idx];

    // ========== 特征 1：多边形顶点数 ==========
    std::vector<cv::Point> approx;
    double epsilon = 0.02 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);
    int vertices = approx.size();

   // ========== 特征 2：凸缺陷 ==========
    std::vector<int> hull_indices;
    cv::convexHull(contour, hull_indices, false, false);
    std::vector<cv::Vec4i> defects; 
    int defect_count = 0;

    try {
        cv::convexityDefects(contour, hull_indices, defects);
        double perimeter = cv::arcLength(contour, true);
        for (auto& d : defects) {
            if (d[3] / 256.0 > perimeter * 0.04)
                defect_count++;
        }
    } catch (const cv::Exception& e) {
        // 自相交轮廓，跳过凸缺陷检测
        defect_count = 0;
    }
    
    // ========== 特征 3：宽高比 ==========
    cv::Rect rect = cv::boundingRect(contour);
    double aspect_ratio = (double)rect.width / rect.height;

    // ========== 特征 4：凸性 ==========
    double area = cv::contourArea(contour);
    std::vector<cv::Point> hull_pts;
    cv::convexHull(contour, hull_pts);
    double hull_area = cv::contourArea(hull_pts);
    double solidity = (hull_area > 0) ? area / hull_area : 0;

    // ========== 特征 5：实心像素密度 ==========
    cv::Mat roi_mask = cv::Mat::zeros(64, 64, CV_8UC1);
    cv::drawContours(roi_mask, contours, max_idx, 255, cv::FILLED);
    cv::Mat filled;
    cv::floodFill(roi_mask, cv::Point(32, 32), 255);
    int filled_pixels = cv::countNonZero(roi_mask);
    double fill_ratio = (hull_area > 0) ?
                        (double)filled_pixels / (64.0 * 64.0) : 0;
    
    if (vertices >= 20 && fill_ratio > 0.4)
    {
        armor_light.righting = 1;
        armor_light.ID = "sentinel";
        return;
    }
    else if (defect_count == 0 && aspect_ratio < 0.5 && vertices <= 8 )
    {
        armor_light.righting = 1;
        armor_light.ID = "1";
        return;
    }
    else if (defect_count >= 1 && aspect_ratio >= 0.4)
    {
        armor_light.righting = 1;
        armor_light.ID = "3";
        return;
    }
    else{
        armor_light.righting = 0;
        armor_light.ID = "UNKNOWN";
        return;
    }
}

int main(){
    int iCameraCounts = 1;
    int iStatues = -1;
    int setStatues = -1;
    tSdkCameraDevInfo   tCameraEnumList;
    double*     pfExposureTime;
    int     hCamera;
    tSdkCameraCapbility     tCapability;
    tSdkFrameHead   sFrameInfo;
    BYTE*       pbyBuffer;
    int     iDisplatFrames = 10000;
    int     channel = 3;

    /////////////////////////////////////////////////////////
    std::vector<lightbors> armor;
    /////////////////////////////////////////////////////////

    CameraSdkInit(1);

    iStatues = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    printf("statue = %d\n", iStatues);

    printf("count = %d\n", iCameraCounts);
    if(iCameraCounts == 0){
        return -1;
    }

    iStatues = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

    printf("state = %d\n", iStatues);
    if(iStatues != CAMERA_STATUS_SUCCESS){
        return -1;
    }

    CameraGetCapability(hCamera, &tCapability);

    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    CameraPlay(hCamera);
    CameraSetAeState(hCamera, false);
    setStatues = CameraSetExposureTime(hCamera, 3000);
    CameraSetGain(hCamera, 100, 70, 50);
    // CameraGetExposureLineTime(hCamera, pfExposureTime);
    // printf("compuse = %lf", *pfExposureTime);
    printf("statue = %d\n", setStatues);

    if(tCapability.sIspCapacity.bMonoSensor){
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }

    while(iDisplatFrames){

        if(CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS){

            CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

            cv::Mat matImage(
					cv::Size(sFrameInfo.iWidth,sFrameInfo.iHeight), 
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
					);

            //      测试图像处理代码
            //       作用为突出数字特征
            cv::Mat grayImage;
            cv::cvtColor(matImage, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat endImage;
            cv::threshold(grayImage, endImage, 4, 255, cv::THRESH_TOZERO);
            cv::threshold(endImage, endImage, 10, 255, cv::THRESH_TOZERO_INV);
            cv::Mat reduceguss;
            cv::GaussianBlur(endImage, reduceguss, cv::Size(5, 5), 0);
            cv::Mat process;
            cv::threshold(reduceguss, process, 10, 0, cv::THRESH_TRUNC);
            cv::Mat end;
            cv::threshold(process, end, 2, 255, cv::THRESH_BINARY);
            cv::Mat end_canny;
            cv::Canny(end, end_canny, 120, 200);
            cv::Mat the_end;

            /*      此内容为灯条检测部分
                    作用为测试灯条检测如何编写
            */
            std::vector<cv::Mat> channels;
            cv::split(matImage, channels);
            cv::Mat r = channels[2];
            cv::Mat mask;
            cv::threshold(r, mask, 150, 255, cv::THRESH_BINARY);
            cv::Mat mask3c;
            cv::cvtColor(mask, mask3c, cv::COLOR_GRAY2BGR);
            std::vector<std::vector<cv::Point>> counters;
            cv::findContours(mask, counters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::RotatedRect> end_rects;

            // 灯条检测逻辑
            for(auto& cnt : counters){
                cv::RotatedRect rotRect = cv::minAreaRect(cnt);

                float area = rotRect.size.width * rotRect.size.height;
                if (area < 50 || area > 9000) continue;
                float width = std::min(rotRect.size.width, rotRect.size.height);
                float height = std::max(rotRect.size.width, rotRect.size.height);
                float ratio = height / width;
                if (ratio < 4 || ratio > 26) continue;
                end_rects.push_back(rotRect);

                cv::Point2f pts[4];
                rotRect.points(pts);
                // for (int i = 0; i < 4; i++)
                // {
                //     cv::line(result, pts[i], pts[(i+1)%4], cv::Scalar(0,255,0), 2);
                // }
                
            }

            armor.clear();
            if(end_rects.size() >= 2){
                // 灯条配对逻辑
                for(size_t i = 0; i < end_rects.size()-1; i++){
                    for(size_t j = i + 1; j < end_rects.size(); j++){

                        /*
                        *****    灯条匹配逻辑      ******
                        */
                        // 倾斜角度偏差检测
                        float angle_TF = end_rects[i].angle - end_rects[j].angle;
                        if (fabs(angle_TF) > 6)continue;
                        
                        // 灯条距离与灯条长度比值检测
                        float first_max = std::max(end_rects[i].size.width, end_rects[i].size.height);
                        float second_max = std::max(end_rects[j].size.width, end_rects[j].size.height);
                        float getheight = end_rects[i].center.x - end_rects[j].center.x;
                        float getlight = (first_max + second_max) / 2;
                        float distance_TF = std::fabs(getheight) / getlight;
                        if (distance_TF > 2.9 || distance_TF < 2.0)continue;

                        /*
                        *****   灯条归位并定点    *****
                        */
                        lightbors armor_light;

                        if(end_rects[i].center.x < end_rects[j].center.x){
                            armor_light.left_lightbors = end_rects[i];
                            armor_light.right_lightbors = end_rects[j];
                        }else{
                            armor_light.left_lightbors = end_rects[j];
                            armor_light.right_lightbors = end_rects[i];
                        }

                        // 计算每对灯条的装甲板
                        set_light(armor_light);

                        // 放入装甲板总数组
                        armor.push_back(armor_light);

                    }
                }

                // 此处为字符识别部分，为了降低算力，采用自研的识别算法
                for(auto& cnt_string : armor){
                    cv::Mat ROI;
                    CropAndResize(cnt_string, end_canny, ROI);

                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(ROI, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    distinguish(cnt_string, contours);
                }

                // 将疑似装甲板全部绘制出来       并配上识别字符
                for(auto& cnt : armor){
                    
                    if(cnt.righting){
                        cv::putText(matImage, cnt.ID, cnt.armor_point[0], cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255,0,0), 3);
                    }
                    for (int i = 0; i < 4; i++)
                    {
                        cv::line(matImage, cnt.armor_point[i], cnt.armor_point[(i+1)%4], cv::Scalar(0,0,254), 2);
                    }
                }
            }

            imshow("Tracking", matImage);
            // imshow("tae", end_canny);

            waitKey(5);

            CameraReleaseImageBuffer(hCamera, pbyBuffer);
        }
    }

    CameraUnInit(hCamera);

    free(g_pRgbBuffer);


    return 0;
}

