/**
 * @brief 串口通信层与通信协议实现
 * @author jokit
 * @date 2026-07-15
 */
#ifndef VISION_SYSTEM_OUTPUT_SERIAL_PORT_HPP
#define VISION_SYSTEM_OUTPUT_SERIAL_PORT_HPP

#include <string>
#include <vector>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif

// amadeus_26 协议常量
constexpr uint8_t FRAME_HEADER_1 = 0xA5;
constexpr uint8_t FRAME_HEADER_2 = 0x5A;
constexpr uint8_t FRAME_TAIL     = 0xFF;
constexpr int FRAME_LENGTH       = 15;

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    // 初始化串口，传入端口号（如 "COM3" 或 "/dev/ttyUSB0"）和波特率
    bool init(const std::string& port_name, int baud_rate = 115200);
    
    // 发送底层字节数据
    bool WriteData(const uint8_t* data, size_t size);
    
    // 断开连接
    void close();

    bool isOpen() const { return is_open_; }

    // 读取底层字节数据 (非阻塞)
    int ReadData(uint8_t* buffer, size_t size);

private:
    bool is_open_ = false;
#ifdef _WIN32
    HANDLE hSerial_;
#else
    int fd_ = -1;
#endif
};

// ==========================================
// 协议发送器
// ==========================================
class ProtocolSender {
public:
    ProtocolSender(SerialPort& serial);
    
    // 发送目标云台偏航角、俯仰角等信息
    void sendTarget(float yaw, float pitch, int16_t feed = 0, uint8_t key = 0);

private:
    SerialPort& serial_;
    int send_cnt_ = 0;
};

// ==========================================
// 协议接收器与坐标转换
// ==========================================
struct GimbalPose {
    float yaw;
    float pitch;
    float roll;
    int64_t timestamp;
};

class ProtocolReceiver {
public:
    ProtocolReceiver(SerialPort& serial);
    
    // 尝试从串口读取并解析一帧电控发来的数据，返回是否成功
    bool receiveGimbalPose(GimbalPose& pose);

private:
    SerialPort& serial_;
    std::vector<uint8_t> rx_buffer_;
};

// 为了解决 "世界坐标系与载车坐标系重合" 的矩阵计算工具
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "vision_system/decision/armor_model.hpp"

class CoordinateTransformer {
public:
    // 计算当前摄像头位于世界(载车)坐标系下的外参位姿矩阵
    static armor_model::Pose calcCameraToWorld(float gimbal_yaw, float gimbal_pitch, const Eigen::Vector3d& cam_offset);
};

#endif // VISION_SYSTEM_OUTPUT_SERIAL_PORT_HPP
