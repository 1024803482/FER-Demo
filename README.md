<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/hci.png"></div>

# CeiLing的人脸表情识别(FER)演示界面
人脸表情识别界面开发：人脸表情是人类情感的重要表达方式，人脸表情识别是人机交互的关键组成部分。
## 环境配置
演示界面基于PyQt5, 人脸检测算法基于face_recognition库，表情识别算法基于Tensorflow2.1 (gpu)，相关库文件安装如下所示：  
  ```
  # 创建fer环境，并配置相关安装包
  conda create -n fer python=3.7
  conda activate fer
  pip install numpy==1.20.0
  pip install tensorflow-gpu==2.1
  pip install imageio
  pip install python-opencv
  pip install face_recognition
  pip install PyQt5
  pip install matplotlib
  ```
## 运行程序
在PyCharm下直接运行camera.py或者在项目路径下运行：
  ```
  conda activate fer
  python camera.py
  ```
## 演示界面展示
演示界面支持实时表情识别、图片识别、视频识别以及独特的“表情模式”，具体如下：
### 初始界面
<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/gui.png" width="50%"></div>

### 实时表情识别
<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/normal.png" width="50%"></div>

<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/sad.png" width="50%"></div>

### 表情模式
<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/happy.png" width="50%"></div>

<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/surprise.png" width="50%"></div>

### 加载图片测试
<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/Level2_result.png" width="50%"></div>

<div align=center><img src="https://github.com/1024803482/FER-Demo/blob/master/images/james_result.png" width="50%"></div>

## 其他
读取中文路径可能存在问题，相关问题可以通过issues或email(cailh@buaa.edu.cn)与我联系😊  
——CeiLing

