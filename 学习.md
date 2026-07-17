# 测试以及学习MindVision摄像头项目
## 前言
*本内容为作者学习使用MindVision摄像头并验证各种算法的测试工具*  
> 系统性能不取决于最强环节，而取决于各部分的协同效率——《工程控制论》 钱学森  
---

## 目录
- [第一节 项目架构简介](#项目架构)
- [第二节 装甲板识别逻辑](#装甲板识别逻辑)
    - [2.1 摄像头配置](#摄像头配置)
    - [2.2 灯条检测](#灯条的检测)
    - [2.3 装甲板检测](#)


## 项目架构
```
test_cxx/
├── assets/
│   ├── camera.yaml     //  摄像机内参文件 （用作位姿解算，目前还没有测试）
|   └── svm_model.xml   //  数字识别权重文件（目前只支持 1 3 sentinel ）
├── src/
│    ├── main.cpp       // 主要执行文件，包含摄像头配置以及灯条检测等内容
│    ├── classify.cpp   // 用于装甲板的数字特征识别，SVM
│    ├── pnpsolver.cpp  // 用于Pnp解算位姿，目前没有验证，只是基于算法进行编写
│    └── test.cpp       // 用于验证装甲板数字识别的实验性文件，如果要使用须在xmake对于的位置进行消除注释，并自行准备好装甲板区域特征图片
└── xmake.lua           // 编译文件........

```

---
## 装甲板识别逻辑


### 摄像头配置
摄像头配置方面，我的调节逻辑为在MindVision官方驱动软件Windows版本进行调节，并记入对应的数值，在main读取时进行配置。

### 灯条的检测及其配对
灯条的检测我有两个**先置条件**：
1：摄像头已经配置好参数，获取的图像就已经处理过了。
```cpp
/**
	* @brief 设置摄像头的曝光以及增益
    * 位于main的301； 
*/
CameraSetAeState(hCamera, false);                          
setStatues = CameraSetExposureTime(hCamera, 5000);
CameraSetGain(hCamera, 100, 70, 50);
```
2.对于图像有如下处理。
```cpp
/**
	* @brief 单独分离出红色通道并进行形态检测
    * 位于main的330； 
    * 实践发现如果加入高斯以及膨胀等形态学处理会导致实际的效果减半：可能原因是将原本的高值进行了加权平均，导致效果下降。
*/
std::vector<cv::Mat> channels;
cv::split(matImage, channels);
cv::Mat r = channels[2];
cv::Mat mask;
cv::threshold(r, mask, 150, 255, cv::THRESH_BINARY);
// cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
// dilate(mask, r, kernel);
std::vector<std::vector<cv::Point>> counters;
cv::findContours(mask, counters, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
std::vector<cv::RotatedRect> end_rects;
```
接下来就是**灯条的检测部分**
```cpp
/**
	* @brief 将灯条进行面积以及长宽比检测，以及合理角度角度检测
    * 位于main的330； 
    * 为了便于控制检测阈值，处理方法为实时进行打印当前灯条的值。
*/
 // 灯条检测逻辑
            for(auto& cnt : counters){
                cv::RotatedRect rotRect = cv::minAreaRect(cnt);

                // 面积检测筛选
                float area = rotRect.size.width * rotRect.size.height;
                if (area < 50 || area > 9000) continue;

                // 灯条比例筛选
                float width = std::min(rotRect.size.width, rotRect.size.height);
                float height = std::max(rotRect.size.width, rotRect.size.height);
                float ratio = height / width;
                if (ratio < 4 || ratio > 15) continue;

                // 灯条合理角度筛选
                if(getright_angle(rotRect) < 5 || getright_angle(rotRect) > 175)continue;

                end_rects.push_back(rotRect);

                cv::Point2f pts[4];
                rotRect.points(pts);
                for (int i = 0; i < 4; i++)
                {
                    cv::line(matImage, pts[i], pts[(i+1)%4], cv::Scalar(0,255,0), 2);
                }
            }
```
接下来就是**灯条的配对部分**
```cpp
/**
	* @brief 将灯条进行合理距离/长度以及偏差角度检测
    * 位于main的369； 
    * 为了便于控制检测阈值，处理方法为实时进行打印当前灯条的值。
*/
if(end_rects.size() >= 2){
    // 灯条配对逻辑
    for(size_t i = 0; i < end_rects.size()-1; i++){
        for(size_t j = i + 1; j < end_rects.size(); j++){
 
            /*
            *****    灯条匹配逻辑      ******
            */
            // 倾斜角度偏差检测
            float angle_TF = getright_angle(end_rects[i]) - getright_angle(end_rects[j]);
            if (fabs(angle_TF) > 6.5)continue;

            // 灯条距离与灯条长度比值检测
            float first_max = std::max(end_rects[i].size.width, end_rects[i].size.height);
            float second_max = std::max(end_rects[j].size.width, end_rects[j].size.height);
            float getheight = sqrt(pow(end_rects[i].center.x - end_rects[j].center.x, 2)+pow(end_rects[i].center.y - end_rects[j].center.y, 2));
            float getlight = (first_max + second_max) / 2;
            float distance_TF = getheight / getlight;
            if (distance_TF > 3.0 || distance_TF < 2.3)continue;

            /*
            *****   灯条归位并定点    *****
            */
            lightbors armor_light;

            if(end_rects[i].center.x < end_rects[j].center.x){
                armor_light.left_lightbors = end_rects[i];
                armor_light.right_lightbors = end_rects[j];
            }else{
                armor_light.left_lightbors = end_rects[j];
                armor_light.right_lightbors = end_rects[i];
            }
```
### 装甲板检测
在上面我们已经完成了灯条的检测及其配对，接下来就是要将装甲板检测出来。

**逻辑**：检测装甲板中间的特征符号。未完待续......
