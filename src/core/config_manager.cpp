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
    det_cfg_.length_max = (float)det["length_max"];
    det_cfg_.length_min = (float)det["length_min"];

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

    cv::FileNode pln = fs["Planner"];
    if (!pln.empty()) {
        planner_cfg_.w_dist = (double)pln["w_dist"];
        planner_cfg_.w_angle = (double)pln["w_angle"];
        planner_cfg_.w_conf = (double)pln["w_conf"];
        planner_cfg_.w_threat = (double)pln["w_threat"];
        planner_cfg_.switch_threshold = (double)pln["switch_threshold"];
        planner_cfg_.min_effective_angle = (double)pln["min_effective_angle"];
        planner_cfg_.max_lost_frames = (int)pln["max_lost_frames"];
        planner_cfg_.max_cov_trace = (double)pln["max_cov_trace"];
        planner_cfg_.max_range = (double)pln["max_range"];
        planner_cfg_.max_normal_angle = (double)pln["max_normal_angle"];
        planner_cfg_.lambda = (double)pln["lambda"];
        planner_cfg_.base_thresh = (double)pln["base_thresh"];
        planner_cfg_.min_lock_frames = (int)pln["min_lock_frames"];
        planner_cfg_.alpha = (double)pln["alpha"];
        planner_cfg_.max_rate = (double)pln["max_rate"];
    } else {
        // Defaults
        planner_cfg_.w_dist = 0.40; planner_cfg_.w_angle = 0.35; planner_cfg_.w_conf = 0.20; planner_cfg_.w_threat = 0.05;
        planner_cfg_.switch_threshold = 0.35; planner_cfg_.min_effective_angle = 0.70;
        planner_cfg_.max_lost_frames = 10; planner_cfg_.max_cov_trace = 100.0; planner_cfg_.max_range = 15.0; planner_cfg_.max_normal_angle = 1.57;
        planner_cfg_.lambda = 0.2; planner_cfg_.base_thresh = 0.05; planner_cfg_.min_lock_frames = 5; planner_cfg_.alpha = 0.3; planner_cfg_.max_rate = 10.0;
    }

    fs.release();
    loaded_ = true;
}
