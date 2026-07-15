#ifndef VISION_SYSTEM_CORE_CONFIG_MANAGER_HPP
#define VISION_SYSTEM_CORE_CONFIG_MANAGER_HPP

#include <string>
#include <opencv2/opencv.hpp>

struct CameraConfig {
    std::string mode;
    std::string video_path;
    int exposure_time;
    int gain_r;
    int gain_g;
    int gain_b;
};

struct DetectorConfig {
    int thresh_val;
    float area_min, area_max;
    float ratio_min, ratio_max;
    float angle_min, angle_max;
    float distance_min, distance_max;
};

struct UIConfig {
    bool show_window;
    bool draw_3d_box;
    bool draw_2d_id;
};

struct SerialConfig {
    std::string port_name;
    int baud_rate;
};

struct ModulesConfig {
    bool enable_detector;
    bool enable_classifier;
    bool enable_pnp;
    bool enable_serial;
};

class ConfigManager {
public:
    ConfigManager(const std::string& file_path);
    bool isLoaded() const { return loaded_; }
    CameraConfig getCameraConfig() const { return cam_cfg_; }
    DetectorConfig getDetectorConfig() const { return det_cfg_; }
    UIConfig getUIConfig() const { return ui_cfg_; }
    SerialConfig getSerialConfig() const { return serial_cfg_; }
    ModulesConfig getModulesConfig() const { return mod_cfg_; }
private:
    bool loaded_ = false;
    CameraConfig cam_cfg_;
    DetectorConfig det_cfg_;
    UIConfig ui_cfg_;
    SerialConfig serial_cfg_;
    ModulesConfig mod_cfg_;
};

#endif
