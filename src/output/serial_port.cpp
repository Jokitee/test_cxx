/**
 * @brief 串口通信层与通信协议实现
 * @author jokit
 * @date 2026-07-15
 */
#include "vision_system/output/serial_port.hpp"
#include "vision_system/core/logger.hpp"
#include <iostream>
#include <time.h>

#define MINMAX(value, min, max) value = (value < min) ? min : ((value > max) ? max : value)

// ==========================================
// SerialPort 基础串口封装 (支持 Windows)
// ==========================================
SerialPort::SerialPort() {
#ifdef _WIN32
    hSerial_ = INVALID_HANDLE_VALUE;
#endif
}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::init(const std::string& port_name, int baud_rate) {
#ifdef _WIN32
    // Windows 下支持大于 COM9 的串口，需加前缀 "\\\\.\\"
    std::string full_port = "\\\\.\\" + port_name;
    hSerial_ = CreateFileA(full_port.c_str(),
                           GENERIC_READ | GENERIC_WRITE,
                           0,
                           NULL,
                           OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL,
                           NULL);

    if (hSerial_ == INVALID_HANDLE_VALUE) {
        Logger::log(LogLevel::ERROR, ErrorCode::SUCCESS, "Failed to open serial port: " + port_name);
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
    
    if (!GetCommState(hSerial_, &dcbSerialParams)) {
        close();
        return false;
    }

    dcbSerialParams.BaudRate = baud_rate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity   = NOPARITY;

    if (!SetCommState(hSerial_, &dcbSerialParams)) {
        close();
        return false;
    }

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;

    SetCommTimeouts(hSerial_, &timeouts);
    is_open_ = true;
    Logger::log(LogLevel::INFO, ErrorCode::SUCCESS, "Serial port opened: " + port_name + " @ " + std::to_string(baud_rate));
    return true;
#else
    Logger::log(LogLevel::ERROR, ErrorCode::SUCCESS, "Linux serial port not implemented yet.");
    return false;
#endif
}

bool SerialPort::WriteData(const uint8_t* data, size_t size) {
    if (!is_open_) return false;
#ifdef _WIN32
    DWORD bytes_written;
    if (!WriteFile(hSerial_, data, size, &bytes_written, NULL)) {
        return false;
    }
    return bytes_written == size;
#else
    return false;
#endif
}

int SerialPort::ReadData(uint8_t* buffer, size_t size) {
    if (!is_open_) return 0;
#ifdef _WIN32
    DWORD bytes_read = 0;
    if (!ReadFile(hSerial_, buffer, size, &bytes_read, NULL)) {
        return 0;
    }
    return bytes_read;
#else
    return 0;
#endif
}

void SerialPort::close() {
#ifdef _WIN32
    if (hSerial_ != INVALID_HANDLE_VALUE) {
        CloseHandle(hSerial_);
        hSerial_ = INVALID_HANDLE_VALUE;
    }
#endif
    is_open_ = false;
}


// ==========================================
// ProtocolSender 拨杆协议实现 (amadeus_26) 发送端
// ==========================================
ProtocolSender::ProtocolSender(SerialPort& serial) : serial_(serial) {}

void ProtocolSender::sendTarget(float yaw, float pitch, int16_t feed, uint8_t key) {
// ... 保持原有代码不变，为节约 token 这里直接衔接后续 ...
    if (!serial_.isOpen()) return;

    // 控制参数
    int16_t x_move = 0;           // 平动左右 [-660, 660]
    int16_t y_move = 0;           // 平动前后 [-660, 660]
    int16_t yaw_val = 0;          // 云台偏航 [-660, 660]
    int16_t pitch_val = 0;        // 云台俯仰 [-660, 660]
    uint8_t left_switch = 3;      // 左拨杆 [1, 3] 2 摩擦轮打开 3 关闭
    uint8_t right_switch = 3;     // 右拨杆 [1, 3] 2 小陀螺 3 关闭

    // 简单帧率统计与控制台输出控制
    static auto last_time = time(nullptr);
    static int fps = 0;
    time_t t = time(nullptr);
    if (last_time != t) {
        last_time = t;
        fps = 0;
    }
    fps += 1;

    // 将角度 (-3, 3) 度 转换为协议范围 [-660, 660]
    yaw_val = static_cast<int16_t>(yaw * 220);
    pitch_val = static_cast<int16_t>(pitch * 220);
    MINMAX(yaw_val, -660, 660);
    MINMAX(pitch_val, -660, 660);

    // 构建数据帧 - amadeus_26
    uint8_t frame[FRAME_LENGTH];
    int idx = 0;

    frame[idx++] = FRAME_HEADER_1;
    frame[idx++] = FRAME_HEADER_2;

    frame[idx++] = x_move & 0xFF;
    frame[idx++] = (x_move >> 8) & 0xFF;

    frame[idx++] = y_move & 0xFF;
    frame[idx++] = (y_move >> 8) & 0xFF;

    frame[idx++] = yaw_val & 0xFF;
    frame[idx++] = (yaw_val >> 8) & 0xFF;

    frame[idx++] = pitch_val & 0xFF;
    frame[idx++] = (pitch_val >> 8) & 0xFF;

    frame[idx++] = feed & 0xFF;
    frame[idx++] = (feed >> 8) & 0xFF;

    // key 参数通过低4位传递能量机关模式信息
    frame[idx++] = (left_switch << 4) | (key & 0x0F);

    // CRC8 (固定为 0xCC)
    frame[idx++] = 0xCC;

    frame[idx++] = FRAME_TAIL;

    serial_.WriteData(frame, sizeof(frame));
    send_cnt_ += 1;
}

// ==========================================
// ProtocolReceiver 接收解析端
// ==========================================
ProtocolReceiver::ProtocolReceiver(SerialPort& serial) : serial_(serial) {}

bool ProtocolReceiver::receiveGimbalPose(GimbalPose& pose) {
    if (!serial_.isOpen()) return false;
    
    uint8_t buf[128];
    int len = serial_.ReadData(buf, sizeof(buf));
    if (len > 0) {
        // 压入环形缓冲区或向量中
        rx_buffer_.insert(rx_buffer_.end(), buf, buf + len);
    }
    
    // 假设电控传上来的云台状态包长度为 8 字节 (根据实际修改):
    // [0] A5 [1] 5A [2..3] Yaw [4..5] Pitch [6] CRC [7] FF
    while (rx_buffer_.size() >= 8) {
        if (rx_buffer_[0] == 0xA5 && rx_buffer_[1] == 0x5A) {
            if (rx_buffer_[7] == 0xFF) { // 校验帧尾
                int16_t yaw_int = (rx_buffer_[3] << 8) | rx_buffer_[2];
                int16_t pitch_int = (rx_buffer_[5] << 8) | rx_buffer_[4];
                
                // 根据下发的 220 比例，逆向解算出云台当前的真实角度
                pose.yaw = yaw_int / 220.0f;
                pose.pitch = pitch_int / 220.0f;
                pose.roll = 0.0f;
                
                // 打上主机时间戳，如果电控传了时间戳可以替换
                auto now = std::chrono::steady_clock::now();
                pose.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
                
                rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 8);
                return true;
            }
        }
        rx_buffer_.erase(rx_buffer_.begin());
    }
    return false;
}

// ==========================================
// 坐标转换工具: 构建相机的世界变换矩阵
// ==========================================
armor_model::Pose CoordinateTransformer::calcCameraToWorld(float gimbal_yaw, float gimbal_pitch, const Eigen::Vector3d& cam_offset) {
    // 角度转弧度
    double y_rad = gimbal_yaw * CV_PI / 180.0;
    double p_rad = gimbal_pitch * CV_PI / 180.0;
    
    // 假设云台坐标系：Z 轴朝前（枪管），X 轴朝右，Y 轴朝下（遵守 OpenCV 习惯）
    // 偏航(Yaw)绕 Y 轴旋转，俯仰(Pitch)绕 X 轴旋转
    Eigen::AngleAxisd yawAngle(y_rad, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(p_rad, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rollAngle(0, Eigen::Vector3d::UnitZ());
    
    // 乘法顺序：根据云台的物理机械结构决定 (通常是先偏航再俯仰)
    Eigen::Quaterniond q_gimbal_to_world = yawAngle * pitchAngle * rollAngle;
    
    armor_model::Pose T_cam_world;
    T_cam_world.linear() = q_gimbal_to_world.matrix();
    // 真实世界坐标 = R * 相机在云台的安装偏置 + 云台原点(此处设为0,0,0)
    T_cam_world.translation() = cam_offset; 
    
    return T_cam_world;
}
