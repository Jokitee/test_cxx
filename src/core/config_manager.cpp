/**
 * @brief YAML配置文件解析与管理器
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/core/config_manager.hpp"
#include "vision_system/core/logger.hpp"

ConfigManager::ConfigManager(const std::string& file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        Logger::log(LogLevel::ERROR, ErrorCode::CONFIG_LOAD_FAILED, "Cannot open " + file_path);
        return;
    }
    
    cv::FileNode cam = fs["Camera"];
    // 如果没有找到 mode 则默认为 camera
    cam_cfg_.mode = cam["mode"].empty() ? "camera" : (std::string)cam["mode"];
    cam_cfg_.video_path = cam["video_path"].empty() ? "" : (std::string)cam["video_path"];
    cam_cfg_.exposure_time = (int)cam["exposure_time"];
    cam_cfg_.gain_r = (int)cam["gain_r"];
    cam_cfg_.gain_g = (int)cam["gain_g"];
    cam_cfg_.gain_b = (int)cam["gain_b"];

    cv::FileNode mod = fs["Modules"];
    if (!mod.empty()) {
        mod_cfg_.enable_detector = (int)mod["enable_detector"] != 0;
        mod_cfg_.enable_classifier = (int)mod["enable_classifier"] != 0;
        mod_cfg_.enable_pnp = (int)mod["enable_pnp"] != 0;
        mod_cfg_.enable_serial = (int)mod["enable_serial"] != 0;
    } else {
        mod_cfg_.enable_detector = true;
        mod_cfg_.enable_classifier = true;
        mod_cfg_.enable_pnp = true;
        mod_cfg_.enable_serial = true;
    }

    cv::FileNode det = fs["Detector"];
    det_cfg_.thresh_val = (int)det["thresh_val"];
    det_cfg_.area_min = (float)det["area_min"];
    det_cfg_.area_max = (float)det["area_max"];
    det_cfg_.ratio_min = (float)det["ratio_min"];
    det_cfg_.ratio_max = (float)det["ratio_max"];
    det_cfg_.angle_min = (float)det["angle_min"];
    det_cfg_.angle_max = (float)det["angle_max"];
    det_cfg_.distance_min = (float)det["distance_min"];
    det_cfg_.distance_max = (float)det["distance_max"];

    cv::FileNode ui = fs["UI"];
    ui_cfg_.show_window = (int)ui["show_window"] != 0;
    ui_cfg_.draw_3d_box = (int)ui["draw_3d_box"] != 0;
    ui_cfg_.draw_2d_id = (int)ui["draw_2d_id"] != 0;
    ui_cfg_.draw_lightbars = (int)ui["draw_lightbars"] != 0;

    cv::FileNode ser = fs["Serial"];
    if (!ser.empty()) {
        serial_cfg_.port_name = (std::string)ser["port_name"];
        serial_cfg_.baud_rate = (int)ser["baud_rate"];
    }

    fs.release();
    loaded_ = true;
}
