import re
from tkinter import messagebox

from PyQt5 import QtWidgets

from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo

from PyQt5.uic.properties import QtCore

from Ui_serial import Ui_MainWindow as Ui_serial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2


font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
from PyQt5 import QtCore


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class Serial(QMainWindow, Ui_serial):
    """ 开始界面 """

    def __init__(self, parent=None):

        super(Serial, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("无线火灾报警系统")     # 设置标题

        # self._num_receive = 0                     # 接收温度数据
        # self._num_receive1 = 0                    # 接收烟雾数据
        self._serial = QSerialPort(self)            # 创建用于连接温度串口的对象
        self._serial1 = QSerialPort(self)           # 创建用于连接烟雾串口的对象
        self._single_slot()                         # 连接信号与槽
        self.get_available_ports()                  # 获取可用的串口列表
        self.set_statusbar()                        # 设置状态栏

        # 设置温度1存储变量，便于后续绘制温度曲线图
        self.tmp1_start_x = 0
        self.tmp1_start_y = 0
        self.tmp1_end_x = 0
        self.tmp1_end_y = 0

        # 设置温度2存储变量，便于后续绘制温度曲线图
        self.tmp2_start_x = 0
        self.tmp2_start_y = 0
        self.tmp2_end_x = 0
        self.tmp2_end_y = 0

        # 设置温度3存储变量，便于后续绘制温度曲线图
        self.tmp3_start_x = 0
        self.tmp3_start_y = 0
        self.tmp3_end_x = 0
        self.tmp3_end_y = 0

        # 设置烟雾传感器1存储变量，便于后续绘制烟雾曲线图
        self.smoke_node1_start_x = 0
        self.smoke_node1_start_y = 0
        self.smoke_node1_end_x = 0
        self.smoke_node1_end_y = 0

        # 设置烟雾传感器2存储变量，便于后续绘制烟雾曲线图
        self.smoke_node2_start_x = 0
        self.smoke_node2_start_y = 0
        self.smoke_node2_end_x = 0
        self.smoke_node2_end_y = 0

        # 设置烟雾传感器3存储变量，便于后续绘制烟雾曲线图
        self.smoke_node3_start_x = 0
        self.smoke_node3_start_y = 0
        self.smoke_node3_end_x = 0
        self.smoke_node3_end_y = 0

        # 温度最大值和标志位的中间变量设置
        self.tmp_max = 0
        self.tmp_flag = 0

        # 烟雾最大值和标志位的中间变量设置
        self.smoke_max = 0
        self.smoke_flag = 0
        self.send_flag = 0
        self.send_senor1 = QtCore.QTimer()

        self.smoke_temp1 = 1
        self.smoke_temp2 = 0
        self.smoke_temp3 = 0

        # 添加图像显示的画布
        # 温度传感器1的画布
        self.static_canvas1 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_3)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas1)  # 向布局groupBox_1中添加渲染器
        # layout.setLabel('left', text='强度', color='#000000')  # y轴设置函数
        self._static_ax1 = self.static_canvas1.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 温度传感器2的画布
        self.static_canvas2 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_4)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas2)  # 向布局groupBox_1中添加渲染器
        self._static_ax2 = self.static_canvas2.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 温度传感器3的画布
        self.static_canvas3 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_5)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas3)  # 向布局groupBox_1中添加渲染器
        self._static_ax3 = self.static_canvas3.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 烟雾传感器1的画布
        self.static_canvas4 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_6)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas4)  # 向布局groupBox_1中添加渲染器
        self._static_ax4 = self.static_canvas4.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 烟雾传感器2的画布
        self.static_canvas5 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_7)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas5)  # 向布局groupBox_1中添加渲染器
        self._static_ax5 = self.static_canvas5.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 烟雾传感器3的画布
        self.static_canvas6 = FigureCanvas(Figure())  # 画布、渲染器
        layout = QtWidgets.QVBoxLayout(self.tmp_graph_8)  # 添加垂直布局类groupBox
        layout.addWidget(self.static_canvas6)  # 向布局groupBox_1中添加渲染器
        self._static_ax6 = self.static_canvas6.figure.subplots(1, 1)  # 从渲染器中的画布figure中，获取子布，也就是Axes（1行1列）

        # 摄像头1和2的定时功能设置
        # self.timer_camera = QtCore.QTimer()
        # self.timer_camera.timeout.connect(self.showvideo1)
        # self.timer_camera1 = QtCore.QTimer()
        # self.timer_camera1.timeout.connect(self.showvideo2)

        # 设置摄像头1和2的标志位
        self.camera1_flag = 0
        self.camera2_flag = 0
        # 设置摄像头存储图片的变量
        self.img = []
        self.img1 = []
        # 设置火灾区域面积大小
        self.area1 = 0
        self.area2 = 0
        # 摄像头1和摄像头2的ip地址设置
        self.ip1 = 0
        self.ip2 = 0
        self.xxx = 0
        self.device = 'cpu'
        # 图片读取进程
        # self.ip1 = 'http://admin:admin@192.168.3.4:8081/'
        self.ip1 = 0
        self.webcam = True
        self.webcam1 = True
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.stopEvent1 = threading.Event()
        self.stopEvent1.clear()
        self.output_size = 480
        self.vid_source =  self.ip1  # 初始设置为摄像头
        self.vid_source1 = self.ip2  # 初始设置为摄像头
        self.model = self.model_load(weights="runs/train/exp_yolov5s/weights/best.pt",
                                     device=self.device)  # todo 指明模型加载的位置的设备

        self.reset_vid()
        self.reset_vid1()
    '''
       ***模型初始化***
       '''

    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    # 信号与槽的连接函数
    def _single_slot(self):
        # self._serial1.readyRead.connect(self.send_data)
        self._serial.readyRead.connect(self.receive_data)                 # 绑定数据读取信号
        self._serial1.readyRead.connect(self.receive_data1)               # 绑定数据读取信号
        self.pushButton_connect.clicked.connect(self.open_close_port)     # 打开关闭温度串口
        self.pushButton_connect_2.clicked.connect(self.open_close_port1)  # 打开关闭烟雾串口
        self.pushButton.clicked.connect(self.send)                        # 发送数据查询烟雾

        # 定时发送烟雾数据
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.send)
        # self.timer_send.timeout.connect(self.send_command_and_receive_data)

        # 发送指令的时间间隔，单位为毫秒
        self.command_interval = 1000
        # 定时接收烟雾数据
        self.tit = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.receive_data1)

        self.pushButton_flush.clicked.connect(self.get_available_ports)    # 刷新1
        self.pushButton_flush_2.clicked.connect(self.get_available_ports)  # 刷新2
        self.save.clicked.connect(self.get_tmp_max)                        # 温度最大值设置
        self.save_2.clicked.connect(self.get_smoke_max)                    # 烟雾最大值设置

        # self.pushButton_2.clicked.connect(self.check_camera_connection0)
        self.pushButton_2.clicked.connect(self.btn_start1)                 # 摄像头1打开和关闭按钮
        # self.pushButton_3.clicked.connect(self.check_camera_connection)
        self.pushButton_3.clicked.connect(self.btn_start2)                 # 摄像头2打开和关闭按钮

    # 设置的温度最大值
    def get_tmp_max(self):
        # 获取键盘输入的最大烟雾值
        temp = self.input_tmp1.text()

        # 验证输入的烟雾值是否为数字
        if re.match("^-?\d+(\.\d+)?$", temp):
            # 如果文本是数字字符串，则将其打印到控制台
            messagebox.showwarning("提示", "温度设置成功")
        else:
            # 如果文本不是数字字符串，弹出一个警告框，并将温度设为0，防止程序报错
            temp = 0
            messagebox.showwarning("警告", "这不是数字格式，请重新输入")

        self.tmp_max = float(temp)
        # 标志位用于判断是否输入温度阈值，方便比较
        self.tmp_flag = 1
        return self.tmp_max, self.tmp_flag

    # 设置的烟雾最大值
    def get_smoke_max(self):
        # 获取键盘输入的最大烟雾值
        smoke = self.input_smoke.text()

        # 验证文本是否为数字
        if re.match("^-?\d+(\.\d+)?$", smoke):
            # 如果文本是数字，则将其打印到控制台
            messagebox.showwarning("提示", "烟雾设置成功")
        else:
            # 如果文本不是数字字符串，弹出一个警告框，并将烟雾设为0，防止程序报错
            smoke = 0
            messagebox.showwarning("警告", "这不是数字格式，请重新输入")

        self.smoke_max = float(smoke)
        # 标志位用于判断是否输入烟雾阈值，方便比较
        self.smoke_flag = 1
        return self.smoke_max, self.smoke_flag

    # 判断摄像头1是否打开和连接
    def check_camera_connection0(self):
        self.cap = cv2.VideoCapture(self.ip1)
        connected = self.cap.isOpened()
        # camera1.release()

        if self.ip1 == 1 & connected:
            messagebox.showwarning("提示", "摄像头2连接成功")
            pass
        else:
            # print("摄像头2未连接")
            messagebox.showwarning("警告", "摄像头1未连接,正在重新连接")

    # 摄像头1的打开和关闭功能
    def btn_start1(self):

        self.vid_source =  self.ip1
        self.webcam = True
        th = threading.Thread(target=self.showvideo1)
        th.start()

    # 检测摄像头2是否打开和连接
    def check_camera_connection(self):
        self.cap2 = cv2.VideoCapture(self.ip2)
        connected1 = self.cap2.isOpened()
        # self.cap2.release()

        if self.ip2 == 0 & connected1:
            # messagebox.showwarning("提示", "摄像头2连接成功")
            pass
        else:
            # print("摄像头2未连接")
            messagebox.showwarning("警告", "摄像头2未连接，正在重新连接")

    def btn_start2(self):
        self.vid_source1 = self.ip2
        self.webcam1 = True
        th1 = threading.Thread(target=self.showvideo2)
        th1.start()

    def showvideo1(self):
        # pass
        model = self.model
        output_size = self.output_size
        # # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)---表示推理时使用的图像尺寸
        conf_thres = 0.25  # confidence threshold---检测结果的置信度（confidence）的阈值
        iou_thres = 0.45  # NMS IOU threshold--NMS中的交并比阈值变量。默认值是0.45
        max_det = 1000  # maximum detections per image--目标检测任务中每张图像最多允许检测到的目标数量
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results--显示结果
        save_txt = False  # save results to *.txt--保存文本
        save_conf = False  # save confidences in --save-txt labels--保存置信度
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)  # 摄像头地址
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
            print(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Write results
            has_flame = False  # 是否检测到火焰
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                        if cls == 0:  # 火焰类别的索引为0
                            # 检测到了火焰
                            has_flame = True
                # 根据检测结果输出
                if has_flame:
                    self.label_30.setText("有火焰")
                else:
                    self.label_30.setText("无火焰")

                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.camera_label2_2.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
            if cv2.waitKey(1) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.reset_vid()
                break


    def reset_vid(self):
        self.camera_label2_2.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source =  self.ip1
        self.webcam = True

    def reset_vid1(self):
        self.camera_label2.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source1 = '0'
        self.webcam1 = True

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()

    def close_vid1(self):
        self.stopEvent1.set()
        self.reset_vid1()


    def showvideo2(self):
        # pass
        model = self.model
        output_size = self.output_size
        # # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)---表示推理时使用的图像尺寸
        conf_thres = 0.25  # confidence threshold---检测结果的置信度（confidence）的阈值
        iou_thres = 0.45  # NMS IOU threshold--NMS中的交并比阈值变量。默认值是0.45
        max_det = 1000  # maximum detections per image--目标检测任务中每张图像最多允许检测到的目标数量
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results--显示结果
        save_txt = False  # save results to *.txt--保存文本
        save_conf = False  # save confidences in --save-txt labels--保存置信度
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source1)  # 摄像头地址
        webcam = self.webcam1
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            has_flame = False
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                        if cls == 0:  # 火焰类别的索引为0
                            # 检测到了火焰
                            has_flame = True
                if has_flame:
                    self.label_32.setText("有火焰")
                else:
                    self.label_32.setText("无火焰")

                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.camera_label2.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))

            if cv2.waitKey(1) & self.stopEvent1.is_set() == True:
                self.stopEvent1.clear()

                self.reset_vid1()
                break

    def play_alert_sound(self):
        # 在这里添加播放警报音频的相关代码
        frequency = 2500  # 警报音频的频率
        duration = 1000  # 警报音频的时长（毫秒）
        winsound.Beep(frequency, duration)

    def set_statusbar(self):
        # 设置状态栏
        self._timer_one_s = QTimer(self)
        self._timer_one_s.timeout.connect(self._update_datetime)
        self._label_datetime = QLabel(self.statusbar)
        self._label_datetime.setText(" " * 30)
        self._label_num_recv = QLabel(self.statusbar)
        self._label_num_recv1 = QLabel(self.statusbar)
        # self._label_num_recv.setText("接收:" + str(self._num_receive))
        # self._label_num_recv1.setText("接收:" + str(self._num_receive1))
        self.statusbar.addPermanentWidget(self._label_num_recv, stretch=0)
        self.statusbar.addPermanentWidget(self._label_num_recv1, stretch=0)
        self.statusbar.addPermanentWidget(self._label_datetime, stretch=0)
        self._timer_one_s.start(500)

    # 更新实时时间
    def _update_datetime(self):
        pass
        # 更新实时时间
        # _real_time = QDateTime.currentDateTime().toString(
        #     " yyyy-MM-dd hh:mm:ss ")
        # self._label_datetime.setText(_real_time)

    # 获取温度和烟雾串口
    def get_available_ports(self):
        # 获取可用的串口
        self._ports = {}  # 用于保存串口的信息
        self.comboBox_port.clear()  # 清空温度串口
        self.comboBox_port_2.clear()  # 清空烟雾串口
        ports = QSerialPortInfo.availablePorts()  # 获取串口信息,返回一个保存串口信息的对象
        if not ports:
            self.statusbar.showMessage("无可用串口!", 5000)
            return
        ports.reverse()  # 逆序
        for port in ports:
            # 通过串口名字关联串口变量,并将其添加到界面控件中
            # print(port.standardBaudRates())
            self._ports[port.portName()] = port
            self.comboBox_port.addItem(port.portName())  # 温度串口选择
            self.comboBox_port_2.addItem(port.portName())  # 烟雾串口选择

    # 打开温度串口
    def open_close_port(self, ):
        # 判断温度串口是否打开
        if self._serial.isOpen():
            # 如果串口处于开启状态
            self._serial.close()
            self.statusbar.showMessage("关闭串口成功", 2000)
            self.pushButton_connect.setText("打开串口")
            self.label_status.setProperty("isOn", False)
            self.label_status.style().polish(self.label_status)  # 刷新样式
            return

        # 获取端口、波特率、校验位、停止位、数据位
        if self.comboBox_port.currentText():
            port = self._ports[self.comboBox_port.currentText()]
            # 设置端口
            self._serial.setPort(port)
            # 设置波特率
            self._serial.setBaudRate(  # QSerialPort::Baud9600
                getattr(QSerialPort,
                        'Baud' + self.comboBox_baud.currentText()))
            # 设置校验位
            self._serial.setParity(  # QSerialPort::NoParity
                getattr(QSerialPort,
                        self.comboBox_parity.currentText() + 'Parity'))
            # 设置数据位
            self._serial.setDataBits(  # QSerialPort::Data8
                getattr(QSerialPort,
                        'Data' + self.comboBox_data.currentText()))
            # 设置停止位
            self._serial.setStopBits(  # QSerialPort::OneStop
                getattr(QSerialPort, self.comboBox_stop.currentText()))

            # NoFlowControl          没有流程控制
            # HardwareControl        硬件流程控制(RTS/CTS)
            # SoftwareControl        软件流程控制(XON/XOFF)
            # UnknownFlowControl     未知控制
            self._serial.setFlowControl(QSerialPort.NoFlowControl)
            # 以读写方式打开串口
            ok = self._serial.open(QIODevice.ReadWrite)
            if ok:
                self.statusbar.showMessage("打开串口成功", 2000)
                self.pushButton_connect.setText('关闭串口')
                self.label_status.setProperty("isOn", True)
                self.label_status.style().polish(self.label_status)  # 刷新样式
            else:
                QMessageBox.warning(self, "警告", "打开串口失败", QMessageBox.Yes)
                # self.statusbar.showMessage('打开串口失败', 2000)
                self.pushButton_connect.setText('打开串口')
                self.label_status.setProperty("isOn", False)
                self.label_status.style().polish(self.label_status)  # 刷新样式
        else:
            QMessageBox.warning(self, "警告", "无可用串口", QMessageBox.Yes)

    # 接收温度数据
    def receive_data(self, ):
        # 接收数据
        num = self._serial.bytesAvailable()
        # print(num)
        if num:
            # self._num_receive += num
            # self._label_num_recv.setText("接收:" + str(self._num_receive))
            _data = self._serial.readAll()

            if self.checkBox_show_current_time.isChecked():
                _real_time = QDateTime.currentDateTime().toString(
                    "yyyy-MM-dd hh:mm:ss")
                real_time = (
                        "<span style='text-decoration:underline; color:green; font-size:12px;'>"
                        + _real_time + "</span>")
                self.textBrowser_receive.append(real_time)
                self.textBrowser_receive.append("")
            if self.radioButton_asciiview.isChecked():
                data = _data.data()
                # 解码显示
                try:
                    self.textBrowser_receive.insertPlainText(
                        data.decode('utf-8'))
                except:  # 解码失败
                    out_s = (
                            r"<span style='text-decoration:underline; color:red; font-size:16px;'>"
                            + r"解码失败" + repr(data) + r"</span>")
                    # self.textBrowser_receive.insertPlainText(out_s)
                    # self.textBrowser_receive.append(out_s)

            elif self.radioButton_hexview.isChecked():
                data = _data.data()
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '0x{:02X}'.format(data[i]) + ' '
                # 将字符串转化为列表
                tmp = out_s.split()
                print(tmp)

                # 计算温度
                hex1 = tmp[8]
                hex2 = tmp[9]

                # 合并十六进制数
                merged_hex = hex(int(hex1, 16) << 8 | int(hex2, 16))
                merged_hex = merged_hex[2:]  # 去掉合并后字符串中的 '0x' 前缀

                # 转换为十进制
                decimal = int(merged_hex, 16)

                # 转换为带有小数点的十进制
                decimal_with_decimal_point = decimal / 10.0

                # 保留两位小数
                real_tmp = round(decimal_with_decimal_point, 2)

                # 显示温度示传感器节点1的数据和图像
                if tmp[4] == '0x01':
                    self.tmp1_end_x += 1
                    self.tmp1_end_y = real_tmp
                    self.label_tmp1.setText(
                        "<font color = red font-size=16px>" + "%.2f" % (real_tmp) + "\n" + "</font>")
                    self._static_ax1.plot([self.tmp1_start_x, self.tmp1_end_x],
                                          [self.tmp1_start_y, self.tmp1_end_y],
                                          'r.-')
                    self.tmp1_start_x = self.tmp1_end_x
                    self.tmp1_start_y = self.tmp1_end_y
                    self.static_canvas1.draw()  # 温度传感器2更新字画布的渲染器

                # 显示温度传感器节点2的数据和图像
                if tmp[4] == '0x01':
                    self.tmp2_end_x += 1
                    self.tmp2_end_y = real_tmp
                    # self.label_tmp2.setText("%.2f" % (real_tmp) + "\n")
                    self.label_tmp3.setText(
                        "<font color = green font-size=16px>" + "%.2f" % (real_tmp) + "\n" + "</font>")
                    self._static_ax2.plot([self.tmp2_start_x, self.tmp2_end_x],
                                          [self.tmp2_start_y, self.tmp2_end_y],
                                          'g.-')
                    self.tmp2_start_x = self.tmp2_end_x
                    self.tmp2_start_y = self.tmp2_end_y
                    self.static_canvas2.draw()  # 温度传感器2更新字画布的渲染器

                # 显示温度传感器节点3的数据和图像
                if tmp[4] == '0x01':
                    self.tmp3_end_x += 1
                    self.tmp3_end_y = real_tmp
                    # self.label_tmp3.setText("%.2f" % (real_tmp) + "\n")
                    self.label_tmp2.setText(
                        "<font color = blue font-size=16px>" + "%.2f" % (real_tmp) + "\n" + "</font>")
                    self._static_ax3.plot([self.tmp3_start_x, self.tmp3_end_x],
                                          [self.tmp3_start_y, self.tmp3_end_y],
                                          'b.-')
                    self.tmp3_start_x = self.tmp3_end_x
                    self.tmp3_start_y = self.tmp3_end_y
                    self.static_canvas3.draw()  # 温度传感器2更新字画布的渲染器

                # 比较温度阈值
                if self.tmp_flag == 1:
                    if self.tmp1_end_y > self.tmp_max:
                        # self.play_alert_sound()
                        self.label_17.setText("<font color = red font-size=16px>" + "温度异常")
                    else:
                        self.label_17.setText("<font color = red font-size=16px>" + "温度正常")
                    if self.tmp2_end_y > self.tmp_max:
                        # self.play_alert_sound()
                        self.label_18.setText("<font color = green font-size=16px>" + "温度异常")
                    else:
                        self.label_18.setText("<font color = green font-size=16px>" + "温度正常")
                    if self.tmp3_end_y > self.tmp_max:
                        # self.play_alert_sound()
                        self.label_19.setText("<font color = purple font-size=16px>" + "温度异常")
                    else:
                        self.label_19.setText("<font color = purple font-size=16px>" + "温度正常")

    # 打开烟雾串口
    def open_close_port1(self, ):
        # 判断烟雾串口是否打开
        if self._serial1.isOpen():
            # 如果串口处于开启状态
            self._serial1.close()
            self.statusbar.showMessage("关闭串口成功", 2000)
            self.pushButton_connect_2.setText("打开串口")
            self.label_status.setProperty("isOn", False)
            self.label_status.style().polish(self.label_status)  # 刷新样式
            return

        if self.comboBox_port_2.currentText():
            port = self._ports[self.comboBox_port_2.currentText()]
            # 设置端口
            self._serial1.setPort(port)
            # 设置波特率
            self._serial1.setBaudRate(  # QSerialPort::Baud9600
                getattr(QSerialPort,
                        'Baud' + self.comboBox_baud_2.currentText()))
            # 设置校验位
            self._serial1.setParity(  # QSerialPort::NoParity
                getattr(QSerialPort,
                        self.comboBox_parity_2.currentText() + 'Parity'))
            # 设置数据位
            self._serial1.setDataBits(  # QSerialPort::Data8
                getattr(QSerialPort,
                        'Data' + self.comboBox_data_2.currentText()))
            # 设置停止位
            self._serial1.setStopBits(  # QSerialPort::OneStop
                getattr(QSerialPort, self.comboBox_stop_2.currentText()))

            # NoFlowControl          没有流程控制
            # HardwareControl        硬件流程控制(RTS/CTS)
            # SoftwareControl        软件流程控制(XON/XOFF)
            # UnknownFlowControl     未知控制
            self._serial1.setFlowControl(QSerialPort.NoFlowControl)
            # 以读写方式打开串口
            self.ok = self._serial1.open(QIODevice.ReadWrite)

            if self.ok:
                self.statusbar.showMessage("打开串口成功", 2000)
                self.pushButton_connect_2.setText('关闭串口')
                self.label_status.setProperty("isOn", True)
                self.label_status.style().polish(self.label_status)  # 刷新样式
            else:
                QMessageBox.warning(self, "警告", "打开串口失败", QMessageBox.Yes)
                self.pushButton_connect_2.setText('打开串口')
                self.label_status.setProperty("isOn", False)
                self.label_status.style().polish(self.label_status)  # 刷新样式
        else:
            QMessageBox.warning(self, "警告", "无可用串口", QMessageBox.Yes)

    # 为烟雾传感器发送指令
    def send(self):
        if self.smoke_temp1 == 1:
            Data1 = '01 03 00 00 00 01 84 0A'
            self.smoke_temp1 = 0
            self.smoke_temp2 = 1
            self.smoke_temp3 = 0
        elif self.smoke_temp2 == 1:
            Data1 = '02 03 00 00 00 01 84 39'
            self.smoke_temp1 = 0
            self.smoke_temp2 = 0
            self.smoke_temp3 = 1
        elif self.smoke_temp3 == 1:
            Data1 = '03 03 00 00 00 01 85 E8'
            self.smoke_temp1 = 1
            self.smoke_temp2 = 0
            self.smoke_temp3 = 0

        if Data1 != "":
            Data1 = Data1.strip()
            send_list = []
            while Data1 != '':
                try:
                    num = int(Data1[0:2], 16)
                except ValueError:
                    return None
                Data1 = Data1[2:].strip()
                send_list.append(num)
            Data1 = bytes(send_list)
            num = self._serial1.write(Data1)
        self.receive_data1()  # 接收烟雾数据

        # 判断是否需要重复发送指令
        # if self.checkBox.isChecked():
        self.timer_send.start(1000)

    # 接收烟雾数据
    def receive_data1(self, ):
        # 接收数据
        num1 = self._serial1.bytesAvailable()
        if num1:
            # self._num_receive1 += num1
            # self._label_num_recv1.setText("接收:" + str(self._num_receive1))

            # print(num1)
            # 当数据可读取时
            # 这里只是简答测试少量数据,如果数据量太多了此处readAll其实并没有读完
            # 需要自行设置粘包协议
            _data = self._serial1.readAll()
            # print("接收数据", _data)

            if self.checkBox_show_current_time.isChecked():
                _real_time = QDateTime.currentDateTime().toString(
                    "yyyy-MM-dd hh:mm:ss")
                real_time = (
                        "<span style='text-decoration:underline; color:green; font-size:12px;'>"
                        + _real_time + "</span>")
                self.textBrowser_receive.append(real_time)
                self.textBrowser_receive.append("")
            if self.radioButton_asciiview_2.isChecked():
                data = _data.data()

                # 解码显示
                try:
                    self.textBrowser_receive.insertPlainText(
                        data.decode('utf-8'))
                except:  # 解码失败
                    out_s = (
                            r"<span style='text-decoration:underline; color:red; font-size:16px;'>"
                            + r"解码失败" + repr(data) + r"</span>")
                    # self.textBrowser_receive.insertPlainText(out_s)
                    # self.textBrowser_receive.append(out_s)

            elif self.radioButton_hexview_2.isChecked():
                data = _data.data()
                # print(data)
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '0x{:02X}'.format(data[i]) + ' '
                # 将字符串转化为列表
                tmp = out_s.split()
                print(tmp)

                # 计算温度和烟雾
                # real_tmp = int(tmp[9], 16) / 10  # 计算温度

                smoke_tmp1 = int(tmp[3], 16)  # 计算烟雾
                smoke_tmp2 = int(tmp[4], 16)
                temp = str(smoke_tmp1) + str(smoke_tmp2)
                # print(temp)
                real_tmp = int(temp)
                # 显示传感器节点1的数据和图像
                if tmp[0] == '0x02':
                    self.smoke_node1_end_x += 1
                    self.smoke_node1_end_y = real_tmp
                    self.label_smoke1.setText("%d" % (real_tmp) + "\n")
                    self._static_ax4.plot([self.smoke_node1_start_x, self.smoke_node1_end_x],
                                          [self.smoke_node1_start_y, self.smoke_node1_end_y],
                                          'r.-')
                    self.smoke_node1_start_x = self.smoke_node1_end_x
                    self.smoke_node1_start_y = self.smoke_node1_end_y
                    self.static_canvas4.draw()  # 烟雾传感器1更新字画布的渲染器

                # 显示传感器节点2的数据和图像
                if tmp[0] == '0x02':
                    self.smoke_node2_end_x += 1
                    self.smoke_node2_end_y = real_tmp
                    self.label_smoke2.setText("%d" % (real_tmp) + "\n")
                    self._static_ax5.plot([self.smoke_node2_start_x, self.smoke_node2_end_x],
                                          [self.smoke_node2_start_y, self.smoke_node2_end_y],
                                          'g.-')
                    self.smoke_node2_start_x = self.smoke_node2_end_x
                    self.smoke_node2_start_y = self.smoke_node2_end_y
                    self.static_canvas5.draw()  # 温度传感器2更新字画布的渲染器

                # 显示传感器节点3的数据和图像
                if tmp[0] == '0x02':
                    self.smoke_node3_end_x += 1
                    self.smoke_node3_end_y = real_tmp
                    self.label_smoke3.setText("%d" % (real_tmp) + "\n")
                    self._static_ax6.plot([self.smoke_node3_start_x, self.smoke_node3_end_x],
                                          [self.smoke_node3_start_y, self.smoke_node3_end_y],
                                          'b.-')
                    self.smoke_node3_start_x = self.smoke_node3_end_x
                    self.smoke_node3_start_y = self.smoke_node3_end_y
                    self.static_canvas6.draw()  # 温度传感器2更新字画布的渲染器

                # # 比较烟雾阈值
                if self.smoke_flag == 1:
                    if self.smoke_node1_end_y < self.smoke_max:
                        self.label_29.setText("烟雾正常")
                    else:
                        self.label_29.setText("烟雾过高")
                        self.play_alert_sound()
                    if self.smoke_node2_end_y < self.smoke_max:
                        self.label_25.setText("烟雾正常")
                    else:
                        self.label_25.setText("烟雾过高")
                        self.play_alert_sound()
                    if self.smoke_node3_end_y < self.smoke_max:
                        self.label_27.setText("烟雾正常")
                    else:
                        self.label_27.setText("温度过高")
                        self.play_alert_sound()

    # 重载关闭窗口事件
    def closeEvent(self, event):
        self._timer_one_s.stop()
        if self._serial.isOpen():
            self._serial.close()

        if self._serial1.isOpen():
            self._serial1.close()
        super(Serial, self).closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    satrtwindow = Serial()
    satrtwindow.show()
    sys.exit(app.exec_())
