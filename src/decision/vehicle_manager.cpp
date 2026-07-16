/**
 * @brief 车辆实体追踪与状态管理器
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/decision/vehicle_manager.hpp"

VehicleNode::VehicleNode(const std::string& id, const armor_model::Camera& cam) 
    : vehicle_id(id), cam_(cam) 
{
}

armor_model::VehicleModel VehicleNode::getVehicleModel() const {
    armor_model::VehicleModel model(0.300, 0.150, 15.0);
    if (tracker) {
        tracker->updateVehicleModel(model);
    }
    return model;
}

void VehicleNode::addArmor(const lightbors& armor) {
    armors_buffer.addPlate(armor);
}

void VehicleNode::processFrame(int64_t timestamp, PoseEstimator& estimator) {
    if (armors_buffer.plates.empty()) {
        is_tracking = false;
        latest_obs.armors.clear();
        return;
    }
    
    latest_obs = armors_buffer.getObservation(timestamp);
    
    // 【核心新增逻辑】：利用车辆历史位姿进行装甲板 ID 的数据关联
    if (is_tracking && tracker_initialized && tracker) {
        armor_model::Pose T_prior = tracker->getCurrentPose(); 
        
        // 提取 3D 模型里四个装甲板的数据
        const auto& plates = getVehicleModel().getPlates();
        std::vector<std::pair<int, cv::Point2f>> projected_plates;
        
        for (const auto& plate : plates) {
            // 计算法向量在相机坐标系下的方向
            armor_model::Vec3 normal_cam = T_prior.linear() * plate.normal_object;
            // 判断可见性：装甲板的法向量朝向相机 (在相机坐标系下，Z轴朝前，如果法向量Z分量为负，说明朝向相机)
            if (normal_cam.z() < 0) {
                // 投影 3D 中心点到 2D
                cv::Point2f pt2d = cam_.project(plate.center_object, T_prior);
                projected_plates.push_back({plate.id, pt2d});
            }
        }
        
        // 修正当前观测的 plate_id
        for (auto& obs_armor : latest_obs.armors) {
            // 计算观测到的 2D 矩形中心点
            cv::Point2f obs_center(0.f, 0.f);
            for (int i = 0; i < 4; i++) {
                obs_center += obs_armor.corners_img[i];
            }
            obs_center.x /= 4.f;
            obs_center.y /= 4.f;
            
            // 在所有可见投影中找最近的一块
            double min_dist = 1e9;
            int best_id = obs_armor.plate_id; // 保底使用默认的排序 ID
            for (const auto& proj : projected_plates) {
                double dist = cv::norm(obs_center - proj.second);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_id = proj.first;
                }
            }
            // 重新分配基于模型预测匹配出的 ID
            obs_armor.plate_id = best_id;
        }
    }
    
    // 如果尚未初始化 tracker，使用当前帧的主装甲板来初始化
    if (!tracker_initialized) {
        const armor_model::ArmorObservation& init_obs = latest_obs.armors[0];
        
        // 我们通过将装甲板放置在其自身的局部坐标系（Z=0），求解出一个大概的初始位姿
        std::vector<cv::Point3f> objPts(4);
        double w = 0.135 / 2.0;
        double h = 0.055 / 2.0;
        // objPts 顺序必须与 imgPts 一致 (top-left, top-right, bottom-right, bottom-left)
        // 在 OpenCV 相机坐标系中，Y向下为正，所以 -h 为 top, +h 为 bottom
        objPts[0] = cv::Point3f(-w, -h, 0); // top-left
        objPts[1] = cv::Point3f( w, -h, 0); // top-right
        objPts[2] = cv::Point3f( w,  h, 0); // bottom-right
        objPts[3] = cv::Point3f(-w,  h, 0); // bottom-left
        
        std::vector<cv::Point2f> imgPts = {
            init_obs.corners_img[0], init_obs.corners_img[1],
            init_obs.corners_img[2], init_obs.corners_img[3]
        };
        
        cv::Vec4d q_wo_tmp; cv::Mat t_wo_tmp, R_wo_tmp;
        if (estimator.estimatePose(objPts, imgPts, q_wo_tmp, t_wo_tmp, R_wo_tmp)) {
            armor_model::Pose T_init = armor_model::Pose::Identity();
            T_init.linear() = armor_model::cvMatToEigen3d(R_wo_tmp);
            T_init.translation() = armor_model::cvMatToEigenVec(t_wo_tmp);
            
            // EKF内部会根据半径r进行补偿，这里直接传入观测到的装甲板位姿
            // T_init.translation() += T_init.linear() * Eigen::Vector3d(0, 0, 0.15);
            
            tracker.reset(new vision_system::VehicleTracker(init_obs, timestamp, T_init));
            tracker_initialized = true;
            is_tracking = true;
        } else {
            is_tracking = false;
        }
    } else {
        // 先进行预测
        tracker->predict(timestamp);
        
        // 遍历所有关联好的装甲板进行观测更新
        for (const auto& obs : latest_obs.armors) {
            std::vector<cv::Point3f> objPts(4);
            double w = 0.135 / 2.0;
            double h = 0.055 / 2.0;
            // objPts 顺序必须与 imgPts 一致 (top-left, top-right, bottom-right, bottom-left)
            objPts[0] = cv::Point3f(-w, -h, 0); // top-left
            objPts[1] = cv::Point3f( w, -h, 0); // top-right
            objPts[2] = cv::Point3f( w,  h, 0); // bottom-right
            objPts[3] = cv::Point3f(-w,  h, 0); // bottom-left
            
            std::vector<cv::Point2f> imgPts = {
                obs.corners_img[0], obs.corners_img[1],
                obs.corners_img[2], obs.corners_img[3]
            };
            
            cv::Vec4d q_tmp; cv::Mat t_tmp, R_tmp;
            if (estimator.estimatePose(objPts, imgPts, q_tmp, t_tmp, R_tmp)) {
                Eigen::Vector3d t_cam_armor = armor_model::cvMatToEigenVec(t_tmp);
                Eigen::Matrix3d R_cam_armor = armor_model::cvMatToEigen3d(R_tmp);
                
                // 使用基于角度的匹配来防止 Target ID Switch
                int best_id = tracker->matchArmor(t_cam_armor, R_cam_armor);
                
                // 根据推断出的真实 ID 修正观测
                const_cast<armor_model::ArmorObservation&>(obs).plate_id = best_id;
                
                tracker->update(best_id, t_cam_armor, R_cam_armor);
            }
        }
        
        current_pose = tracker->getCurrentPose();
        is_tracking = true;
    }
    
    // 3. 清理当前帧缓存，准备接收下一帧的消息订阅
    armors_buffer.plates.clear();
}


VehicleManager::VehicleManager(const armor_model::Camera& cam, PoseEstimator& estimator) 
    : am_cam_(cam), estimator_(estimator) {}

void VehicleManager::update(const std::vector<lightbors>& armors, int64_t timestamp) {
    // 1. 将检测到的有效装甲板按车辆 ID 分发
    for (const auto& armor : armors) {
        if (!armor.righting) continue;
        if (armor.ID == "unknown" || armor.ID == "unknow") continue;
        
        std::string id = armor.ID;
        // 如果车辆实体尚不存在，则在管理器中进行实例化注册
        if (nodes_.find(id) == nodes_.end()) {
            nodes_.emplace(id, VehicleNode(id, am_cam_));
        }
        nodes_.at(id).addArmor(armor);
    }
    
    // 2. 发送信号驱动所有在册的车辆节点开始当前帧的位姿解算和模型迭代
    for (auto& pair : nodes_) {
        pair.second.processFrame(timestamp, estimator_);
    }
}
