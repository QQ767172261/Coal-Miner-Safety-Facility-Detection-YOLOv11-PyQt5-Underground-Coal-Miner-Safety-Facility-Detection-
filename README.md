矿井下煤矿工人安全设施检测数据集，4369张（视频抽帧抽取），提供yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/22bd3e81b2174107ac0b22c61c949eef.png)

图像尺寸:640*640
类别数量:4类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69043e6305ea4e8b9af105a0d45d6fb1.png)

训练集图像数量:3059; 验证集图像数量:875； 测试集图像数量:435
类别名称: 每一类图像数 ，每一类标注数
person - 人员：4170, 10341
helmet - 安全帽：3876, 10652
self-rescuer - 自救器：2940, 4522
indicator - 指示器：2451, 6185
image num: 4369
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3ce5f383fed34ea6b7f36596927c2a6e.png)


2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e1ea01c851c4c99aab6cccfef9bc0b5.png)

3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4318df87217b429d9d964f2d8a384c6b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96e9278725c644ae93fce2d7ba1bc9b1.png)
# 煤矿工人安全设施检测 YOLOv11 + PyQt5 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b53a1a6646054ffb9346211139a1b20b.png)

## 1、数据集配置文件 coal_config.yaml
```yaml
path: ./datasets/coal_mine
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
  2: self‑rescuer
  3: indicator
```

## 2、训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolov11n轻量化模型
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="coal_config.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )
    print("训练完成")
```

## 3、PyQt5可视化推理界面 main_gui.py
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QLabel, QFileDialog, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class CoalDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11煤矿工人安全设施检测系统")
        self.resize(1280,720)
        self.model=YOLO("./runs/detect/train/weights/best.pt")

        #控件
        self.btn_img=QPushButton("图片检测",self)
        self.btn_img.move(30,20)
        self.btn_img.clicked.connect(self.detect_image)

        self.btn_video=QPushButton("视频检测",self)
        self.btn_video.move(130,20)
        self.btn_video.clicked.connect(self.detect_video)

        self.btn_cam=QPushButton("摄像头实时检测",self)
        self.btn_cam.move(230,20)
        self.btn_cam.clicked.connect(self.detect_camera)

        self.label_show=QLabel(self)
        self.label_show.setGeometry(20,80,800,600)

        self.log_text=QTextEdit(self)
        self.log_text.setGeometry(850,80,400,600)

    def detect_image(self):
        path,_=QFileDialog.getOpenFileName(self,"打开图片","","Image(*.jpg *.png *.jpeg)")
        if not path:
            return
        res=self.model.predict(source=path,save=False)
        img=res[0].plot()
        h,w,c=img.shape
        bytes_per_line = c*w
        q_img=QImage(img.data,w,h,bytes_per_line,QImage.Format_RGB888).rgbSwapped()
        self.label_show.setPixmap(QPixmap.fromImage(q_img).scaled(self.label_show.size(),Qt.KeepAspectRatio))

        info=""
        for box in res[0].boxes:
            cls_name=self.model.names[int(box.cls)]
            conf=float(box.conf)
            x1,y1,x2,y2=map(int,box.xyxy[0])
            info+=f"类别:{cls_name} 置信度:{conf:.2f} 坐标:xmin={x1},ymin={y1},xmax={x2},ymax={y2}\n"
        self.log_text.setText(info)

    def detect_video(self):
        path,_=QFileDialog.getOpenFileName(self,"打开视频","","Video(*.mp4 *.avi)")
        if not path:
            return
        cap=cv2.VideoCapture(path)
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:break
            res=self.model.predict(frame,save=False)
            frame=res[0].plot()
            cv2.imshow("视频检测",frame)
            if cv2.waitKey(1)&0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_camera(self):
        cap=cv2.VideoCapture(0)
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret:break
            res=self.model.predict(frame,save=False)
            frame=res[0].plot()
            cv2.imshow("摄像头实时检测",frame)
            if cv2.waitKey(1)&0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=CoalDetectWindow()
    win.show()
    sys.exit(app.exec_())
```
## 运行依赖安装
```bash
pip install ultralytics pyqt5 opencv-python
```

