import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QTextEdit, QLabel, QFileDialog, QDialog, \
    QDialogButtonBox, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
from PIL import Image
import torch
from torchvision import transforms
from model import *
import os

class HistoryWindow(QDialog):
    def __init__(self, history_list):
        super().__init__()

        self.setWindowTitle('历史记录')
        self.setGeometry(1000, 500, 1000, 800)  # 设置窗口大小
        self.setStyleSheet("background-color: #2F4F4F; color: #FFFFFF;")  # 深色背景和白色文字

        layout = QGridLayout()

        self.history_listwidget = QListWidget()
        self.history_listwidget.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        layout.addWidget(self.history_listwidget, 0, 0, 1, 2)  # 放在网格的第0行，第0列，跨1行2列

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        button_box.setStyleSheet("background-color: #3CB371;")
        layout.addWidget(button_box, 1, 1)  # 放在网格的第1行，第1列

        # 将历史记录添加到列表中
        for item in history_list:
            self.history_listwidget.addItem(QListWidgetItem(item))

        self.setLayout(layout)

class HandwrittenDigitRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        directory = "data"
        subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        self.indexed_folders = {}
        index = 0
        for subdirectory in subdirectories:
            self.indexed_folders[index] = subdirectory
            index += 1

        self.model = ResNet18(num_classes=26)
        checkpoint = torch.load('./weights/ResNet18.pth', map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('水果分类及新鲜度')
        self.setGeometry(100, 100, 1800, 900)  # 设置窗口大小为 1920x1080
        self.setStyleSheet("background-color: #2F4F4F; color: #FFFFFF;")  # 设置深色背景和白色文字

        layout = QGridLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(1200, 1000)  # 调整标签的大小，以适应更大的窗口
        self.image_label.setStyleSheet("border: 2px solid #3CB371;")  # 绿色边框

        # 设置窗口的大小
        windowGeometry = self.geometry()
        windowGeometry.setWidth(1200)
        windowGeometry.setHeight(1400)
        self.setGeometry(windowGeometry)

        # 将窗口居中于屏幕
        screenGeometry = QApplication.desktop().availableGeometry()
        self.move((screenGeometry.width() - windowGeometry.width()) / 2,
                  (screenGeometry.height() - windowGeometry.height()) / 2)

        upload_button = QPushButton('上传图片')
        upload_button.setIcon(QIcon("icons/upload.png"))  # 假设有一个上传的图标
        upload_button.clicked.connect(self.upload_image)
        upload_button.setStyleSheet("background-color: #3CB371; color: #FFFFFF; font-size: 32px;")

        recognize_button = QPushButton('识别图片')
        recognize_button.setIcon(QIcon("icons/recognize.png"))  # 假设有一个识别的图标
        recognize_button.clicked.connect(self.recognize_digit)
        recognize_button.setStyleSheet("background-color: #3CB371; color: #FFFFFF; font-size: 32px;")

        history_button = QPushButton('分类历史')
        history_button.setIcon(QIcon("icons/history.png"))  # 假设有一个历史记录的图标
        history_button.clicked.connect(self.show_history)
        history_button.setStyleSheet("background-color: #3CB371; color: #FFFFFF; font-size: 32px;")

        self.history_textedit = QTextEdit()
        self.history_textedit.setReadOnly(True)
        self.history_textedit.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")
        self.history_textedit.setFont(QFont("Arial", 14))  # 调整字体大小以适应更大的窗口

        self.history_list = []  # 用于保存历史记录的列表

        # 将控件添加到网格布局中
        layout.addWidget(self.image_label, 0, 0, 1, 3)  # 第0行，第0列，跨1行3列
        layout.addWidget(upload_button, 1, 0)  # 第1行，第0列
        layout.addWidget(recognize_button, 1, 1)  # 第1行，第1列
        layout.addWidget(history_button, 1, 2)  # 第1行，第2列
        layout.addWidget(self.history_textedit, 2, 0, 1, 3)  # 第2行，第0列，跨1行3列

        self.setLayout(layout)

    def upload_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '打开图片', '', '图像文件 (*.png *.jpg *.bmp)')

        if image_path:
            pixmap = QPixmap(image_path)
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            scaled_pixmap = pixmap.scaled(label_width, label_height, aspectRatioMode=Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = image_path

    def recognize_digit(self):
        if hasattr(self, 'current_image_path'):
            image = Image.open(self.current_image_path)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image = transform(image).unsqueeze(0)  # 添加批次维度

            with torch.no_grad():
                output = self.model(image)

            predicted_digit = torch.argmax(output).item()
            file_name = os.path.basename(self.current_image_path)
            history_item = f"图片：{file_name}\n预测结果：{self.indexed_folders[predicted_digit]}\n"
            self.history_list.append(history_item)
            self.history_textedit.append(history_item)

    def show_history(self):
        history_window = HistoryWindow(self.history_list)
        history_window.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置应用程序风格为Fusion
    main_window = HandwrittenDigitRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())
