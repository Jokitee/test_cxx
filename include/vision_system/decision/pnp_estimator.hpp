#ifndef VISION_SYSTEM_DECISION_PNP_ESTIMATOR_HPP
#define VISION_SYSTEM_DECISION_PNP_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>
#include <string>

// Pnp解算类
class PoseEstimator {
public:
    PoseEstimator() = default;
    explicit PoseEstimator(const std::string& yamlPath);

    bool loadCameraYAML(const std::string& path);
    void setWorldPose(const cv::Mat& R_wc, const cv::Mat& t_wc);
    bool estimatePose(cv::InputArray objPts,
                      cv::InputArray imgPts,
                      cv::Vec4d&    q_wo,
                      cv::Mat&      t_wo,
                      cv::Mat&      R_wo) const;

    static cv::Vec4d rotMatToQuat(const cv::Mat& R);

    const cv::Mat& cameraMatrix()    const { return K_; }
    const cv::Mat& distCoeffs()      const { return dist_; }
    const cv::Mat& worldRotation()   const { return R_wc_; }
    const cv::Mat& worldTranslation()const { return t_wc_; }
    int imageWidth()                 const { return width_; }
    int imageHeight()                const { return height_; }

private:
    cv::Mat K_;
    cv::Mat dist_;
    int     width_  = 0;
    int     height_ = 0;

    cv::Mat R_wc_;
    cv::Mat t_wc_;

    bool intrinsicsLoaded_ = false;
    bool extrinsicsSet_    = false;
};

#endif // VISION_SYSTEM_DECISION_PNP_ESTIMATOR_HPP
