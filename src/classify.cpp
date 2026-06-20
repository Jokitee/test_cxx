	#include <opencv2/opencv.hpp>
	#include <iostream>
	#include <vector>
	#include <string>
	// 必须与 Python 端训练代码完全一致
	const cv::Size WIN_SIZE(48, 36);
	const cv::Size BLOCK_SIZE(12, 12);
	const cv::Size BLOCK_STRIDE(6, 6);
	const cv::Size CELL_SIZE(6, 6);
	const int NBINS = 9;
	// 类别名称表（需与 Python 端保持顺序一致）
	const std::vector<std::string> CLASS_NAMES = {"1", "2", "3"};
	/**
	 * @brief 对传入的 Mat 图像进行 HOG 特征提取和 SVM 预测
	 * @param input_img 输入的原始图像 (BGR 格式)
	 * @param hog 初始化好的 HOGDescriptor 对象
	 * @param svm 加载好的 SVM 模型指针
	 * @return std::string 识别出的类别名称，如果失败返回 "unknown"
	 */
	std::string detectImage(const cv::Mat& input_img, cv::HOGDescriptor& hog, cv::Ptr<cv::ml::SVM>& svm) {
	    if (input_img.empty()) {
	        return "unknown";
	    }
	    // 1. 预处理（必须与训练时完全一致）
	    cv::Mat img;
	    cv::resize(input_img, img, WIN_SIZE);
	    cv::Mat gray;
	    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	    cv::equalizeHist(gray, gray); // 光照归一化
	    // 2. 提取 HOG 特征
	    std::vector<float> descriptors;
	    hog.compute(gray, descriptors);
	    if (descriptors.empty()) {
	        return "unknown";
	    }
	    // 将一维 vector 转换为 1行 x N列 的 CV_32F Mat 矩阵
	    cv::Mat feat(1, descriptors.size(), CV_32FC1, descriptors.data());
	    // 3. SVM 预测
	    float result = svm->predict(feat);
	    int predicted_label_id = static_cast<int>(result);
	    // 4. 映射为类别字符串
	    if (predicted_label_id >= 0 && predicted_label_id < static_cast<int>(CLASS_NAMES.size())) {
	        return CLASS_NAMES[predicted_label_id];
	    }
	    return "unknown";
	}
	// ================= 使用示例 =================
	int main() {
	    std::string model_path = "svm_model.xml";
	    std::string test_image_path = "datatest/test1.jpg";
	    // 1. 加载 SVM 模型
	    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(model_path);
	    if (svm.empty()) {
	        std::cerr << "错误: 无法加载模型文件 " << model_path << std::endl;
	        return -1;
	    }
	    // 2. 初始化 HOG 描述子 (只需初始化一次，重复传入函数即可)
	    cv::HOGDescriptor hog(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS);
	    // 3. 读取一张图片 (或者从摄像头/视频流中获取的 Mat)
	    cv::Mat frame = cv::imread(test_image_path);
	    if (frame.empty()) {
	        std::cerr << "错误: 无法读取图片 " << test_image_path << std::endl;
	        return -1;
	    }
	    // 4. 调用检测函数
	    std::string predicted_class = detectImage(frame, hog, svm);
	    std::cout << "图片识别结果: " << predicted_class << std::endl;
	    // 可视化展示
	    cv::putText(frame, "Class: " + predicted_class, cv::Point(10, 30), 
	                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
	    cv::imshow("Detection", frame);
	    cv::waitKey(0);
	    return 0;
	}