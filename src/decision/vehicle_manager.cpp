/**
 * @brief 车辆实体追踪与状态管理器
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/decision/vehicle_manager.hpp"

VehicleNode::VehicleNode(const std::string& id, const armor_model::Camera& cam) 
    : vehicle_id(id), 
      // 每个车辆都有自己专属的模型优化器，初始模型假设 300mm 宽、150mm 高、倾角 15 度
      optimizer(cam, armor_model::VehicleModel(0.300, 0.150, 15.0)),
      cam_(cam) 
{
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
    if (is_tracking && ekf_initialized) {
        armor_model::Pose T_prior = current_pose; 
        
        // 提取 3D 模型里四个装甲板的数据
        const auto& plates = optimizer.getModel().getPlates();
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
    
    // 1. 获取 PnP 初始猜测值
    armor_model::Pose T_init = armor_model::Pose::Identity();
    bool has_init_pose = false;

    // 直接使用最新观测到的主装甲板提供初始 PnP 猜测
    const armor_model::ArmorObservation& init_obs = latest_obs.armors[0];
    
    // 【核心修复】：直接从当前车辆模型中获取对应装甲板在“车辆世界坐标系(载车坐标系)”下的真实3D坐标
    // 这样 solvePnP 算出来的直接是 T_cam_object（相机到车辆中心），而不是 T_cam_armor（相机到单块装甲板）！
    const armor_model::ArmorPlate* plate_model = optimizer.getModel().getPlateById(init_obs.plate_id);
    
    if (plate_model != nullptr) {
        std::vector<cv::Point3f> objPts(4);
        for (int i = 0; i < 4; i++) {
            // 注意：corners_object 单位为米
            objPts[i] = cv::Point3f(
                static_cast<float>(plate_model->corners_object[i].x()),
                static_cast<float>(plate_model->corners_object[i].y()),
                static_cast<float>(plate_model->corners_object[i].z())
            );
        }

        std::vector<cv::Point2f> imgPts = {
            init_obs.corners_img[0], init_obs.corners_img[1],
            init_obs.corners_img[2], init_obs.corners_img[3]
        };
        
        cv::Vec4d q_wo_tmp; cv::Mat t_wo_tmp, R_wo_tmp;
        if (estimator.estimatePose(objPts, imgPts, q_wo_tmp, t_wo_tmp, R_wo_tmp)) {
            Eigen::Matrix3d R_eigen = armor_model::cvMatToEigen3d(R_wo_tmp);
            Eigen::Vector3d t_eigen = armor_model::cvMatToEigenVec(t_wo_tmp);
            
            T_init.linear() = R_eigen;
            // 因为 objPts 单位现在已经是米，solvePnP 返回的 t_eigen 自然也是米，直接赋值，无需 /1000
            T_init.translation() = t_eigen; 
            has_init_pose = true;
        }
    }

    // 2. 将当前帧观测数据送入该车辆独立的 Optimizer 进行迭代优化 
    // (模型参数会随着帧数增加被逐渐合理化)
    if (has_init_pose) {
        // 【核心控制】：只有当看到 2 块及以上的装甲板时，才能约束求解车辆结构参数 (d, h, tilt)
        // 否则如果只有 1 块装甲板，这是一个欠定方程（8个已知数求9个未知数），只能拟合位姿
        bool can_optimize_model = (latest_obs.armors.size() >= 2);
        
        latest_opt_result = optimizer.optimizeSingleFrame(
            latest_obs, 
            T_init, 
            can_optimize_model, // optimize_d (车辆半径)
            can_optimize_model, // optimize_h (装甲板高度差)
            can_optimize_model  // optimize_tilt (装甲板倾角)
        );
        
        current_pose = latest_opt_result.T_cam_object;
        
        // ----------------------------------------------------
        // EKF 卡尔曼滤波与卡方检验 (实时补偿)
        // ----------------------------------------------------
        Eigen::Vector3d obs_pos = current_pose.translation();
        Eigen::VectorXd z(3);
        z << obs_pos.x(), obs_pos.y(), obs_pos.z();
        
        if (!ekf_initialized) {
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);
            x0.head(3) = z; // 初始位置
            ekf.init(x0, timestamp);
            last_timestamp = timestamp;
            ekf_initialized = true;
        } else {
            double dt = (timestamp - last_timestamp) / 1000.0; // 假设 timestamp 是毫秒
            if (dt > 0.0) {
                // 1. 预测
                ekf.predict(dt);
                
                // 2. 检验卡方值，比较预测与当前观测的偏差
                double chi2 = ekf.update(z);
                
                // 3. 实时补偿逻辑：
                // 如果突变过大 (卡方值爆炸，例如突然跳变超过 50)，可能是检测到了另一台车，或是严重干扰
                if (chi2 > 50.0) {
                    // 放弃跟踪，强制重置滤波器到当前位置
                    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(6);
                    x0.head(3) = z;
                    ekf.reset(x0);
                } else {
                    // 取 EKF 平滑和预测过后的最优估计结果作为真实的 3D 坐标
                    Eigen::VectorXd x_opt = ekf.getState();
                    current_pose.translation() = Eigen::Vector3d(x_opt(0), x_opt(1), x_opt(2));
                }
            }
            last_timestamp = timestamp;
        }
        
        is_tracking = true;
    } else {
        is_tracking = false;
        // 长时间丢失后可重置 EKF
        ekf_initialized = false; 
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
