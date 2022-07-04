import sys
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QFileDialog, QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QIcon, QImage
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from DenseNet121 import *
import imageio

labeldict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Normal'}
labelchinese = {0:'生气', 1:'厌恶', 2:'害怕', 3:'高兴', 4:'伤心', 5:'惊讶', 6:'平淡'}
inputs = keras.Input(shape=(48, 48, 1), batch_size=64)
x = create_dense_net(7, inputs, include_top=True, depth=121, nb_dense_block=4, growth_rate=16, nb_filter=-1,
                     nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5, dropout_rate=0.2,
                     activation='softmax')
model = tf.keras.Model(inputs, x, name='densenet121')
filepath = 'DenseNet121.h5'
model.load_weights(filepath)


class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QTimer()  # camera 多线程显示
        self.timer_video = QTimer() # video 多线程显示

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0

        self.image = None

        self.setWindowTitle('Ceiling的表情识别小屋')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap('background.png')))
        self.setPalette(palette)
        self.setWindowIcon(QIcon('icon.ico'))

        self.set_ui()
        self.slot_init()

    def set_ui(self):
        self.mainlayout = QVBoxLayout()
        self.showlayout = QHBoxLayout()
        self.buttonlayout = QVBoxLayout()

        self.button_open_camera = QPushButton('开启摄像头')
        self.button_exit = QPushButton('退出程序')
        self.button_emotion = QPushButton('表情模式')
        self.button_open_file = QPushButton('读取文件')

        self.emotion_module = False

        button_color = [self.button_open_camera, self.button_exit, self.button_open_file, self.button_emotion]
        for i in range(4):
            button_color[i].setStyleSheet("QPushButton{color:rgb(100,220,220)}"
                                          "QPushButton:hover{color:rgb(220,100,100)}"
                                          "QPushButton:hover{font-size:13pt}"
                                          #"QPushButton{background-color:rgb(100,220,220)}"
                                          "QPushButton{font-family:arial }"
                                          "QPushButton{font-weight:bold}"
                                          "QPushButton{font-size:12pt}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:2px 4px}")

        self.button_open_camera.setFixedSize(130, 25)
        self.button_exit.setFixedSize(130, 25)
        self.button_open_file.setFixedSize(130,25)
        self.button_emotion.setFixedSize(130, 25)

        self.label_camera = QLabel()
        self.label_camera.setFixedSize(641, 481)
        self.label_camera.setAutoFillBackground(False)
        self.label_camera.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("'background.png'")
        self.label_camera.setPixmap(pixmap)

        self.label_show_message = 'Tips:\n欢迎来到测试平台\n摄像头尚未开启'
        self.label_show = QLabel(self.label_show_message)
        self.label_show.setAlignment(Qt.AlignTop|Qt.AlignLeft)
        self.label_show.setFixedSize(400, 160)
        self.label_show.setWordWrap(True)
        self.label_show.setStyleSheet("QLabel{color:rgb(250,50,50)}"
                                      "QLabel{font-size:12pt}"
                                      "QLabel{font-family:arial}"
                                      "QLabel{font-weight:bold}"
                                      "QLabel{border:2px}")

        self.buttonlayout.addWidget(self.button_open_camera)
        self.buttonlayout.addWidget(self.button_open_file)
        self.buttonlayout.addWidget(self.button_emotion)
        self.buttonlayout.addWidget(self.button_exit)

        self.showlayout.addWidget(self.label_show)
        self.showlayout.addLayout(self.buttonlayout)

        self.mainlayout.addWidget(self.label_camera, 0, Qt.AlignCenter)
        self.mainlayout.addLayout(self.showlayout, 0)

        self.setLayout(self.mainlayout)
        self.setFixedSize(800, 700)

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_open_file.clicked.connect(self.open_file)
        self.timer_video.timeout.connect(self.show_video)
        self.button_emotion.clicked.connect(self.change_emotion_module)
        self.button_exit.clicked.connect(self.close)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                msg = QMessageBox.warning(self, 'Warning', '请检查摄像机与电脑是否正确连接',
                                          buttons=QMessageBox.OK,
                                          defaultButton=QMessageBox.OK)
                self.label_show_message = 'Tips:\n摄像头连接出现问题'
                self.label_show.setText(self.label_show_message)
            else :
                self.timer_video.stop()
                self.button_open_file.setText('打开文件')
                self.timer_camera.start(30)
                self.button_open_camera.setText('关闭摄像头')
                self.label_show.setStyleSheet("QLabel{color:rgb(250,50,50)}"
                                              "QLabel{font-size:12pt}"
                                              "QLabel{font-family:arial}"
                                              "QLabel{font-weight:bold}"
                                              "QLabel{border:2px}"
                                              )

        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_camera.clear()
            self.label_camera.setAutoFillBackground(False)
            self.label_camera.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap("'background.png'")
            self.label_camera.setPixmap(pixmap)
            self.button_open_camera.setText('开启摄像头')
            self.label_show_message = 'Tips:\n摄像头已关闭'
            self.label_show.setText(self.label_show_message)
            self.label_show.setStyleSheet("QLabel{color:rgb(250,50,50)}"
                                          "QLabel{font-size:12pt}"
                                          "QLabel{font-family:arial}"
                                          "QLabel{font-weight:bold}"
                                          "QLabel{border:2px}")


    def show_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.label_show_message = 'Tips:\n摄像头已开启'
            if self.emotion_module:
                self.label_show_message = self.label_show_message + "\n表情模式已开启"
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            frame, faces, locations = face_detect(frame)
            if faces is not None:
                for i in range(len(faces)):
                    top, right, bottom, left = locations[i]
                    face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48, 48))
                    face = face / 255.0
                    num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                    label = labeldict[num]
                    frame = cv2.putText(frame, label, (left, top), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 1, cv2.LINE_AA)
                    if i <= 3:
                        self.label_show_message = self.label_show_message + '\n人物表情{}：'.format(i + 1) + labelchinese[num]
                    if self.emotion_module:
                        frame = face_replace(frame, locations[i], label)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.label_camera.setPixmap(QPixmap.fromImage(showImage))
            self.label_show.setText(self.label_show_message)

    def open_file(self):
        if not self.timer_video.isActive():
            openfile_name, _ = QFileDialog.getOpenFileName(self, '选择文件')
            suffix = openfile_name.split('.')[-1] # 后缀名
            if suffix == 'png' or suffix == 'jpg' or suffix == 'bmp' or suffix == 'tiff':
                print(openfile_name)
                image = imageio.imread(openfile_name)
                frame = cv2.resize(image, (640, 480))
                if image.ndim == 3:
                    self.label_show_message = 'Tips:\n成功打开图片' + openfile_name
                    if self.emotion_module:
                        self.label_show_message = self.label_show_message + "\n表情模式已开启"
                    frame, faces, locations = face_detect(frame)
                    if faces is not None:
                        for i in range(len(faces)):
                            top, right, bottom, left = locations[i]
                            face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                            face = cv2.resize(face, (48, 48))
                            face = face / 255.0
                            num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                            label = labeldict[num]
                            frame = cv2.putText(frame, label, (left, top), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 1,
                                                cv2.LINE_AA)
                            if i<=3:
                                self.label_show_message = self.label_show_message + '\n人物表情{}：'.format(i+1) + labelchinese[num]
                            if self.emotion_module:
                                frame = face_replace(frame, locations[i], label)
                    image = frame.copy()
                    showImage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    self.label_camera.setPixmap(QPixmap.fromImage(showImage))
                else :
                    self.label_show_message = 'Tips:\n成功打开图片' + openfile_name
                    if self.emotion_module:
                        self.label_show_message = self.label_show_message + "\n表情模式已开启"
                    frame, faces, locations = face_detect(frame)
                    if faces is not None:
                        for i in range(len(faces)):
                            top, right, bottom, left = locations[i]
                            face = cv2.resize(faces[i], (48, 48))
                            face = face / 255.0
                            num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                            label = labeldict[num]
                            frame = cv2.putText(frame, label, (left, top), cv2.FONT_ITALIC, 0.8, (255), 1,
                                                cv2.LINE_AA)
                            if i <= 3:
                                self.label_show_message = self.label_show_message + '\n人物表情{}：'.format(i+1) + labelchinese[num]
                            if self.emotion_module:
                                frame = face_replace(frame, locations[i], label, 0)
                    self.label_camera.setPixmap(QPixmap.fromImage(frame))

                self.label_show.setText(self.label_show_message)
                self.label_show.setStyleSheet("QLabel{color:rgb(250,50,50)}"
                                              "QLabel{font-size:12pt}"
                                              "QLabel{font-family:arial}"
                                              "QLabel{font-weight:bold}"
                                              "QLabel{border:2px}")

            elif suffix == 'avi' or suffix == 'mp4':
                self.cap.open(openfile_name)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, 'Warning', '视频打开失败',
                                        buttons=QMessageBox.Ok,
                                        defaultButton=QMessageBox.Ok)
                else:
                    self.timer_camera.stop()
                    self.button_open_camera.setText('开启摄像头')
                    self.timer_video.start(30)
                    self.button_open_file.setText('关闭视频')
                    self.label_show_message = 'Tips:\n成功打开视频文件:\n'+openfile_name
                    self.label_show.setText(self.label_show_message)
            elif len(openfile_name) == 0:
                self.label_show_message = 'Tips:\n取消打开文件'
                self.label_show.setText(self.label_show_message)
            else :
                QMessageBox.warning(self, 'Warning', '{}文件打开失败，请打开视频或者图像文件'.format(openfile_name),
                                    buttons=QMessageBox.Ok,
                                    defaultButton=QMessageBox.Ok)
                self.label_show_message = 'Tips:\n文件打开失败'
                self.label_show.setText(self.label_show_message)

        else:
            self.timer_video.stop()
            self.cap.release()
            self.label_camera.clear()
            pixmap = QPixmap("'background.png'")
            self.label_camera.setPixmap(pixmap)
            self.button_open_file.setText('打开文件')
            self.label_show_message = '<b>Tips:</b> <br></br>视频放映结束'
            self.label_show.setText(self.label_show_message)


    def show_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.label_show_message = 'Tips:\n播放视频文件'
            if self.emotion_module:
                self.label_show_message = self.label_show_message + "\n表情模式已开启"
            frame = cv2.resize(frame, (640, 480))
            frame, faces, locations = face_detect(frame)
            if faces is not None:
                for i in range(len(faces)):
                    top, right, bottom, left = locations[i]
                    face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48, 48))
                    face = face / 255.0
                    num = np.argmax(model.predict(np.reshape(face, (-1, 48, 48, 1))))
                    label = labeldict[num]
                    frame = cv2.putText(frame, label, (left, top), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 1, cv2.LINE_AA)
                    if i <= 3:
                        self.label_show_message = self.label_show_message + '\n人物表情{}：'.format(i + 1) + labelchinese[num]
                    if self.emotion_module:
                        frame = face_replace(frame, locations[i], label)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
            self.label_show.setText(self.label_show_message)
            self.label_camera.setPixmap(QPixmap.fromImage(showImage))

    def change_emotion_module(self):
        self.emotion_module = not self.emotion_module
        if self.emotion_module == True:
            self.button_emotion.setText('退出模式')
        else:
            self.button_emotion.setText('表情模式')

    def closeEvent(self, event):
        ok = QPushButton()
        cancel = QPushButton()
        msg = QMessageBox(QMessageBox.Warning, '退出程序', '是否退出程序?')
        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cancel, QMessageBox.RejectRole)
        ok.setText('确定')
        cancel.setText('取消')
        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(App.exec_())
