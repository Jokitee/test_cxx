#ifndef VISION_SYSTEM_DECISION_TRACKER_HPP
#define VISION_SYSTEM_DECISION_TRACKER_HPP

#include "vision_system/core/types.hpp"
#include "vision_system/decision/armor_model.hpp"
#include <vector>
#include <string>
#include <algorithm>

// 车辆装甲板类：将同一车辆检测到的装甲板聚合在一起
class VehicleArmors {
public:
    std::string vehicle_id;          // 车辆装甲板数字/名称（如 "1", "2" 等）
    std::vector<lightbors> plates;   // 属于该车辆的装甲板集合

    VehicleArmors() = default;
    explicit VehicleArmors(const std::string& id) : vehicle_id(id) {}

    // 录入装甲板
    void addPlate(const lightbors& armor) {
        if (armor.ID == vehicle_id || vehicle_id.empty()) {
            if (vehicle_id.empty()) vehicle_id = armor.ID;
            plates.push_back(armor);
        }
    }

    // 根据 armor_model 的命名规则生成观测数据，匹配装甲板的 plate_id
    armor_model::FrameObservation getObservation(int64_t timestamp = 0) const {
        armor_model::FrameObservation obs;
        obs.timestamp = timestamp;

        std::vector<lightbors> sorted_plates = plates;
        std::sort(sorted_plates.begin(), sorted_plates.end(), [](const lightbors& a, const lightbors& b) {
            return a.armor_center.x < b.armor_center.x;
        });

        for (size_t i = 0; i < sorted_plates.size(); ++i) {
            armor_model::ArmorObservation a_obs;
            
            if (sorted_plates.size() == 1) {
                a_obs.plate_id = 0; 
            } else if (sorted_plates.size() == 2) {
                if (i == 0) a_obs.plate_id = 3; 
                if (i == 1) a_obs.plate_id = 0; 
            } else {
                a_obs.plate_id = i % 4; 
            }

            a_obs.confidence = 1.0;
            for (int j = 0; j < 4; j++) {
                a_obs.corners_img[j] = sorted_plates[i].armor_point[j];
            }
            obs.armors.push_back(a_obs);
        }
        return obs;
    }
};

#endif // VISION_SYSTEM_DECISION_TRACKER_HPP
