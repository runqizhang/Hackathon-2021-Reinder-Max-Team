# -*- coding: utf-8 -*-
import threading
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from mainWindow import Ui_mainWindow
from Test import Ui_Test
import dlib  # 人脸识别的库dlib
import cv2  # 图像处理的库OpenCv
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import time
import math
from playsound import playsound
import img_rc


class testWindow(QWidget, Ui_Test):
    returnSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(testWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.camera_on)
        self.label_eyeMin.setText('0.225')
        self.label_eyeMax.setText('0.28')
        self.label_mouthMin.setText('0.7')
        self.label_mouthMax.setText('0.7')
        self.label_img.setPixmap(QPixmap(":/camera.png"))
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                      [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                      [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                      [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                      [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                      [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                      [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                      [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                      [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                      [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                      [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                      [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                      [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                      [0.000000, -7.415691, 4.070434]])  # 6下巴角
        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)
        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]
        self.count = 0
        self.eyeMin = 0.235
        self.eyeMax = 0.235
        self.mouthMin = 0.7
        self.mouthMax = 0.7

    def save(self):
        pass

    def get_head_pose(self, shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示
        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        return reprojectdst, euler_angle  # 投影误差，欧拉角

    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self, event):
        global eyeMin, eyeMax, mouthMin, mouthMax
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("68_face_landmarks.dat")
        # 分别获取左右眼面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() == True:  # 返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
        else:
            # 显示封面图
            self.label_img.setPixmap(QPixmap(":/camera.png"))
        # 成功打开视频，循环读取视频流
        mouthMaxList = []
        mouthMinList = []
        eyeMaxList = []
        eyeMinList = []
        if self.cap.isOpened() == True:
            self.label_tip.setText('重复以下动作，努力张大嘴巴，然后闭上嘴巴')
            time.sleep(0.8)
            while (self.count < 100):
                time.sleep(0.05)
                # 返回两个值：
                #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
                flag, im_rd = self.cap.read()  # 图像对象，图像的三维矩阵
                img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  # 取灰度
                faces = self.detector(img_gray, 0)  # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
                if (len(faces) != 0):  # 如果检测到人脸
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                        # 将脸部特征信息转换为数组array的格式
                        shape = face_utils.shape_to_np(shape)
                        mouth = shape[mStart:mEnd]  # 嘴巴坐标
                        mar = self.mouth_aspect_ratio(mouth)  # 打哈欠
                        if mar < self.mouthMin:
                            mouthMinList.append(mar)
                        if mar > self.mouthMin:
                            mouthMaxList.append(mar)
                        mouthHull = cv2.convexHull(mouth)  # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
                height, width = im_rd.shape[:2]
                show = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.label_img.setPixmap(QPixmap.fromImage(showImage))
                self.count += 1
            mouthMinList = sorted(mouthMinList)
            mouthMaxList = sorted(mouthMaxList, reverse=True)
            mouthMinFinal = mouthMinList[0:10]
            mouthMaxFinal = mouthMaxList[0:10]
            average_mouthMin = np.mean(mouthMinFinal)
            average_mouthMax = np.mean(mouthMaxFinal)
            self.label_mouthMin.setText(str('%.4f' % (float(average_mouthMin))))
            self.label_mouthMax.setText(str('%.4f' % (float(average_mouthMax))))
            self.label_tip.setText('重复以下动作，努力睁大眼睛，然后慢慢闭上眼睛')
            time.sleep(0.8)
            print(self.count)
            while (100 <= self.count <= 250):
                time.sleep(0.05)
                # 返回两个值：
                #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
                flag, im_rd = self.cap.read()  # 图像对象，图像的三维矩阵
                img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  # 取灰度
                faces = self.detector(img_gray, 0)  # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
                if (len(faces) != 0):  # 如果检测到人脸
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                        # 将脸部特征信息转换为数组array的格式
                        shape = face_utils.shape_to_np(shape)
                        # 同理，判断是否打哈欠
                        # if mar > 0.75:  # 张嘴阈值0.5
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        # 循环，满足条件的，眨眼次数+1
                        if ear < self.eyeMin:
                            eyeMinList.append(ear)
                        if ear > self.eyeMax:
                            eyeMaxList.append(ear)
                        print(ear)
                        # 点头
                        reprojectdst, euler_angle = self.get_head_pose(shape)
                        har = euler_angle[0, 0]  # 取pitch旋转角度
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        for start, end in self.line_pairs:
                            cv2.line(im_rd, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                        else:
                            pass
                        # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
                        height, width = im_rd.shape[:2]
                show = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.label_img.setPixmap(QPixmap.fromImage(showImage))
                self.count += 1
            eyeMinList = sorted(eyeMinList)
            eyeMaxList = sorted(eyeMaxList, reverse=True)
            eyeMinFinal = eyeMinList[0:10]
            eyeMaxFinal = eyeMaxList[0:10]
            average_eyeMin = np.mean(eyeMinFinal)
            average_eyeMax = np.mean(eyeMaxFinal)
            self.label_eyeMin.setText(str('%.4f' % (float(average_eyeMin))))
            self.label_eyeMax.setText(str('%.4f' % (float(average_eyeMax))))
            self.cap.release()
            self.label_img.setPixmap(QPixmap(':/camera.png'))  # 释放摄像头
            self.label_tip.setText('测试完成，点击右上角退出')

    windowList = []

    def closeEvent(self, QCloseEvent):
        global eyeMin, eyeMax, mouthMin, mouthMax
        eyeMin = self.label_eyeMin.text()
        eyeMax = self.label_eyeMax.text()
        mouthMin = self.label_mouthMin.text()
        mouthMax = self.label_mouthMax.text()
        f = open('userdata.txt', 'w')
        Text = str(mouthMin) + '\n' + str(mouthMax) + '\n' + str(eyeMin) + '\n' + str(eyeMax)
        f.write(Text)
        f.close()
        ui = myWindow()
        self.windowList.append(ui)
        self.close()
        ui.show()

    def camera_on(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))

    def off(self, event):
        """关闭摄像头，显示封面页"""
        self.label_img.setPixmap(QPixmap(":/camera.png"))
        try:
            self.cap.release()
            self.cap.release()
        except:
            pass


class myWindow(QWidget, Ui_mainWindow):
    returnSignal = pyqtSignal()
    windowList = []

    def __init__(self, parent=None):
        super(myWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowOpacity(85)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.setWindowFlags(Qt.FramelessWindowHint)
        global eyeMin, eyeMax, mouthMin, mouthMax

        self.pushButton_start.clicked.connect(self.camera_on)
        self.pushButton_save.clicked.connect(self.save)
        self.pushButton_stop.clicked.connect(self.off)
        self.spinBox_fatiguetime.textChanged.connect(self.AR_CONSEC_FRAMES)
        self.spinBox_offjobtime.textChanged.connect(self.OUT_AR_CONSEC_FRAMES)  # 脱岗时间设置
        self.checkBox_nodding.setChecked(True)
        self.checkBox_wink.setChecked(True)
        self.checkBox_yawn.setChecked(True)
        self.checkBox_offjob.setChecked(True)
        self.spinBox_fatiguetime.setValue(3)
        self.spinBox_offjobtime.setValue(3)
        self.pushButton_quit.clicked.connect(self.exit)
        self.label_img.setPixmap(QPixmap(":/camera.png"))
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        # 闪烁阈值（秒）
        self.sleepLimit = 3
        self.OJ_Limit = 3
        """计数"""
        # 初始化帧计数器和眨眼总数
        self.CloseEyeCounter = 0
        self.CloseEyeNum = 0
        # 初始化帧计数器和打哈欠总数
        self.openMouthCounter = 0
        self.YawnNum = 0
        # 初始化帧计数器和点头总数
        self.NodCounter = 0
        self.NodNum = 0
        # 离职时间长度
        self.OJCounter = 0
        self.OJStatus = 0
        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                      [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                      [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                      [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                      [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                      [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                      [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                      [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                      [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                      [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                      [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                      [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                      [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                      [0.000000, -7.415691, 4.070434]])  # 6下巴角
        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)
        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]
        f = open('userdata.txt', 'r+')
        data = f.read().split('\n')
        print(data)
        mouthMin = float(data[0])
        mouthMax = float(data[1])
        eyeMin = float(data[2])
        eyeMax = float(data[3])

    def exit(self):
        self.quit()

    def save(self):
        try:
            saveurl, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'save', './',
                                                               'Text files(*.txt)')
            f = open(saveurl, 'w+', )
            list = []
            for i in range(self.listWidget_status.count()):
                text = self.listWidget_status.item(i).text()
                list.append(text)
            for j in list:
                f.write(str(j) + '\n')
        except:
            pass

    def get_head_pose(self, shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示
        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        return reprojectdst, euler_angle  # 投影误差，欧拉角

    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self, event):
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("68_face_landmarks.dat")
        self.listWidget_status.addItem("Import Model Success!!!")
        self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
        # 分别获取左右眼面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() == True:  # 返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
            self.listWidget_status.addItem(u"Open Camera Success!!!")
            self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
        else:
            self.listWidget_status.addItem("Open Camera Failed!!!")
            self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
            # 显示封面图
            self.label_img.setPixmap(QPixmap(":/camera.png"))
        # 成功打开视频，循环读取视频流
        while (self.cap.isOpened()):
            time.sleep(0.1)
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            flag, im_rd = self.cap.read()  # 图像对象，图像的三维矩阵
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)  # 取灰度
            faces = self.detector(img_gray, 0)  # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
            if (len(faces) != 0):  # 如果检测到人脸
                self.OJCounter = 0
                self.OJStatus = 0
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(im_rd, d)
                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    # 将脸部特征信息转换为数组array的格式
                    shape = face_utils.shape_to_np(shape)
                    if self.checkBox_yawn.isChecked() == True:  # 打哈欠
                        mouth = shape[mStart:mEnd]  # 嘴巴坐标
                        mar = self.mouth_aspect_ratio(mouth)  # 打哈欠
                        mouthHull = cv2.convexHull(mouth)  # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                        # 同理，判断是否打哈欠
                        if mar > (mouthMin + mouthMax) / 2:  # 张嘴阈值0.5
                            self.openMouthCounter += 1
                        else:
                            # 如果连续3次都小于阈值，则表示打了一次哈欠
                            if self.openMouthCounter >= self.sleepLimit:  # 阈值：3
                                self.YawnNum += 1
                                # 显示
                                cv2.putText(im_rd, "Yarming!", (200, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                                self.listWidget_status.addItem(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"Yarming")
                                self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
                            # 重置嘴帧计数器
                            self.openMouthCounter = 0
                        cv2.putText(im_rd, "Faces: {}".format(len(faces)), (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "CloseEye: {}".format(self.CloseEyeNum), (140, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Yawn: {}".format(self.YawnNum), (300, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Nod: {}".format(self.NodNum), (450, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (255, 255, 0), 2)
                    else:
                        pass
                    # 闭眼
                    if self.checkBox_wink.isChecked() == True:
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        print(ear)
                        print(eyeMin)
                        print(eyeMax)
                        # 循环，满足条件的，眨眼次数+1
                        if ear < eyeMin or ear > eyeMax:  # 眼睛长宽比：0.2
                            self.CloseEyeCounter += 1
                            if self.CloseEyeCounter >= 10:  # 阈值：3g
                                self.CloseEyeNum += 1
                                self.listWidget_status.addItem(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"CloseEye")
                                self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
                                # 重置眼帧计数器
                                self.CloseEyeCounter = 0

                            if self.CloseEyeNum > 5:
                                self.CloseEyeNum = 0
                                t = threading.Thread(target=self.playsound, args=(event,))  # 创建一个线程
                                t.start()
                        else:
                            self.CloseEyeCounter = 0
                        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                        cv2.putText(im_rd, "Faces: {}".format(len(faces)), (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "CloseEye: {}".format(self.CloseEyeNum), (140, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Yawn: {}".format(self.YawnNum), (300, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Nod: {}".format(self.NodNum), (450, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (255, 255, 0), 2)

                    else:
                        pass
                    # 点头
                    if self.checkBox_nodding.isChecked() == True:
                        # 获取头部姿态
                        reprojectdst, euler_angle = self.get_head_pose(shape)
                        har = euler_angle[0, 0]  # 取pitch旋转角度
                        if har > 0.3:  # 点头阈值0.3
                            self.NodCounter += 1
                        else:
                            # 如果连续3次都小于阈值，则表示瞌睡点头一次
                            if self.NodCounter >= 5:  # 阈值：3
                                self.NodNum += 1
                                self.listWidget_status.addItem(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"Nodding")
                                if self.NodNum > 15:
                                    self.NodNum = 0
                                    t = threading.Thread(target=self.playsound, args=(event,))  # 创建一个线程
                                    t.start()
                            # 重置点头帧计数器
                            self.NodCounter = 0
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        for start, end in self.line_pairs:
                            cv2.line(im_rd, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                        # 显示角度结果
                        cv2.putText(im_rd, "Faces: {}".format(len(faces)), (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "CloseEye: {}".format(self.CloseEyeNum), (140, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Yawn: {}".format(self.YawnNum), (300, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Nod: {}".format(self.NodNum), (450, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (255, 255, 0), 2)
                    else:
                        pass

            else:
                if self.OJStatus == 0:
                    self.listWidget_status.addItem(
                        time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"Off Job!!!")
                    self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
                self.OJStatus = 1
                # 没有检测到人脸
                self.OJCounter += 1
                cv2.putText(im_rd, "No Face", (240, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if self.OJCounter > self.OJ_Limit * 10:
                    t3 = threading.Thread(target=self.playsound, args=(event,))  # 创建一个线程
                    t3.start()
                    self.OJCounter = 0

            # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头30次
            # if self.CloseEyeNum >= 50 or self.YawnNum >= 15 or self.NodNum >= 30:
            #     cv2.putText(im_rd, "SLEEP!!!", (110, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 3)
            #     t2 = threading.Thread(target=self.playsound, args=(event,))  # 创建一个线程
            #     t2.start()
            # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height, width = im_rd.shape[:2]
            show = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.label_img.setPixmap(QPixmap.fromImage(showImage))
        # 释放摄像头
        self.cap.release()
        self.label_img.setPixmap(QPixmap(":/camera.png"))

    def camera_on(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))

    def AR_CONSEC_FRAMES(self):
        num = self.spinBox_fatiguetime.text()
        self.listWidget_status.addItem(u"set Fatigue Time: " + num + "s")
        self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
        self.sleepLimit = int(num)

    def OUT_AR_CONSEC_FRAMES(self):
        num = self.spinBox_offjobtime.text()
        self.listWidget_status.addItem(u"set Off-Job Time: " + num + "s")
        self.listWidget_status.setCurrentRow(self.listWidget_status.count() - 1)
        self.OJ_Limit = int(num)

    def off(self, event):
        """关闭摄像头，显示封面页"""
        self.label_img.setPixmap(QPixmap(":/camera.png"))
        try:
            self.cap.release()
            self.cap.release()
        except:
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        try:
            if Qt.LeftButton and self.m_flag:
                self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
                QMouseEvent.accept()
        except:
            pass

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def quit(self):
        self.close()

    def playsound(self, event):
        playsound('bee.mp3')
        self.CloseEyeNum = 0
        self.YawnNum = 0
        self.NodNum = 0
        self.OJCounter = 0

    def OnClose(self):
        """关闭窗口事件函数"""
        info = QMessageBox.information(self, 'Warn', 'Are you sure you want to turn off the camera?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if info == 'ok':
            self.exit()


if __name__ == "__main__":
    import sys

    eyeMin = 0
    eyeMax = 0
    mouthMin = 0
    mouthMax = 0
    app = QtWidgets.QApplication(sys.argv)
    f = open('userdata.txt', 'a+')
    data = f.read().split('\n')
    print(data)
    try:
        if data[0] == '':
            ui = testWindow()
            ui.show()
        else:
            data = f.read().split('\n')
            mouthMin = float(data[0])
            mouthMax = float(data[1])
            eyeMin = float(data[2])
            eyeMax = float(data[3])
            ui = myWindow()
            ui.show()
    except:
        print('error')
    sys.exit(app.exec_())
