/**
 * @brief 3D立体包围框渲染工具
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/output/visualizer.hpp"

void drawCube(
    cv::Mat& image,
    const cv::Mat& R_wo,
    const cv::Mat& t_wo,
    const cv::Mat& K,
    const cv::Mat& dist)
{
    static std::vector<cv::Point3f> cube3d;
    std::vector<cv::Point3f> pts_cam;
    std::vector<cv::Point2f> pts2d;
    
    if (cube3d.empty()) {
        float h = 125.0f, w = 135.0f, l = 30.0f;
        cube3d = {
        {-w/2,-h/2,0}, {w/2,-h/2,0}, {w/2,h/2,0}, {-w/2,h/2,0},
        {-w/2,-h/2,l}, {w/2,-h/2,l}, {w/2,h/2,l}, {-w/2,h/2,l}
        };
    }
    
    pts_cam.resize(8);
    pts2d.resize(8);
    
    const double* R = R_wo.ptr<double>();
    const double* t = t_wo.ptr<double>();
    
    for (int i = 0; i < 8; ++i) {
        const auto& p = cube3d[i];
        double x = R[0]*p.x + R[1]*p.y + R[2]*p.z + t[0];
        double y = R[3]*p.x + R[4]*p.y + R[5]*p.z + t[1];
        double z = R[6]*p.x + R[7]*p.y + R[8]*p.z + t[2];
        pts_cam[i] = cv::Point3f(static_cast<float>(x),
                                  static_cast<float>(y),
                                  static_cast<float>(z));
    }
    
    cv::projectPoints(pts_cam, cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                      K, dist, pts2d);

    int edges[][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    for (auto& e : edges) {
        cv::line(image, pts2d[e[0]], pts2d[e[1]],
                 cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
}
