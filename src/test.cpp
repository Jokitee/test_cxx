#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

struct lightbors
{
    std::string ID;

};

class FeatureDetector8Classes {
private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::HOGDescriptor hog;
    cv::Size imgSize = cv::Size(48, 36); // 归一化尺寸，需与训练时一致
    
    // 映射表：SVM输出的索引 0~7 对应您的8个特征名称
    std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "6", "7", "sentinel"};
 
public:
    FeatureDetector8Classes() {
        // 初始化HOG参数（低算力优选参数）
        hog.winSize = imgSize;
        hog.blockSize = cv::Size(12, 12);
        hog.blockStride = cv::Size(6, 6);
        hog.cellSize = cv::Size(6, 6);
        hog.nbins = 9;
        
        // 加载训练好的SVM模型
        svm = cv::ml::SVM::load("asset/svm_8classes_model.xml");
        if (svm.empty()) {
            std::cout <<  "模型加载失败" << svm.empty() << std::endl; 
        }
    }
 
    /**
     * 检测特征并返回特征名称
     * @param inputImg 输入的局部特征图像 (BGR格式)
     * @return 返回特征名称，失败返回 "unknown"
     */
    std::string detect(const cv::Mat& inputImg) {
        if (inputImg.empty()) return "unknown";
 
        // ================= 1. 图像归一化处理 =================
        cv::Mat normalizedImg;
        // 尺寸归一化
        resize(inputImg, normalizedImg, imgSize, 0, 0, cv::INTER_LINEAR);
        // 色彩归一化（转灰度降算力）
        if (normalizedImg.channels() == 3) {
            cvtColor(normalizedImg, normalizedImg, cv::COLOR_BGR2GRAY);
        }
        // 光照归一化（抗光照干扰）
        equalizeHist(normalizedImg, normalizedImg);
        // 数值归一化（映射到0~1）
        normalizedImg.convertTo(normalizedImg, CV_32F, 1.0 / 255.0);
        // ====================================================
 
        // ================= 2. 特征提取 =================
        std::vector<float> descriptors;
        hog.compute(normalizedImg, descriptors);
        cv::Mat featureMat(1, descriptors.size(), CV_32FC1, descriptors.data());
 
        // ================= 3. 分类预测 =================
        cv::Mat response;
        svm->predict(featureMat, response);
        
        // 获取SVM预测的整数索引 (0~7)
        int predicted_label = static_cast<int>(response.at<float>(0, 0));
        
        // 将索引映射为指定的名称并返回
        if (predicted_label >= 0 && predicted_label < 8) {
            return class_names[predicted_label];
        }
        return "unknown";
    }
};

int main(){

    lightbors armor;
    cv::Mat image = cv::imread("asset/1.jpg", cv::IMREAD_COLOR);
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::vector<std::vector<cv::Point>> counters;
    cv::Mat Canny_P;
    cv::Canny(gray, Canny_P, 50, 150);
    cv::findContours(Canny_P, counters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat result = image.clone();
    std::cout << "ID:" << armor.ID << std::endl;

    cv::imshow("Display", result);
    cv::waitKey(0);

}