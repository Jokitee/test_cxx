/**
 * @brief ONNX神经网络字符分类器
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/decision/classifier.hpp"
#include <iostream>

FeatureDetectorONNX::FeatureDetectorONNX() {
    try {
        net_ = cv::dnn::readNetFromONNX("asset/tiny_resnet.onnx");
        if (net_.empty()) {
            std::cerr << "ONNX 模型加载失败" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "ONNX 模型加载异常: " << e.what() << std::endl;
    }
}

std::string FeatureDetectorONNX::detect(const cv::Mat& inputImg, lightbors& armor_light) {
    if (inputImg.empty()) return "unknown";

    cv::Mat gray;
    if (inputImg.channels() == 3) {
        cv::cvtColor(inputImg, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = inputImg.clone();
    }

    cv::Mat input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
    double scale = std::min(32.0 / gray.cols, 32.0 / gray.rows);
    int h = static_cast<int>(gray.rows * scale);
    int w = static_cast<int>(gray.cols * scale);
    
    if (h == 0 || w == 0) {
        armor_light.righting = false;
        return "unknown";
    }
    
    cv::Rect roi(0, 0, w, h);
    cv::resize(gray, input(roi), cv::Size(w, h));

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max_val, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;

    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    int label_id = label_point.x;

    if (confidence > 0.5 && label_id >= 0 && label_id < 8) {
        armor_light.righting = true;
        return class_names[label_id];
    }

    armor_light.righting = false;
    return "unknown";
}
