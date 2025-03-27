import sys
import os
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QProgressBar, 
                           QVBoxLayout, QGraphicsDropShadowEffect, QMainWindow, 
                           QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QPoint, QEasingCurve
from PyQt5.QtGui import (QPixmap, QPainter, QColor, QLinearGradient, QPen, 
                        QFont, QPainterPath, QBrush, QRadialGradient)
from logwindows import save_log, InfoLog

# 设置控制台输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        # 设置窗口基本属性
        self.setWindowTitle('启动中')
        self.setFixedSize(580, 420)  # 缩小到原来的一半
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 初始化粒子效果
        self.particles = []
        for _ in range(50):
            self.particles.append({
                'x': random.randint(0, self.width()),
                'y': random.randint(0, self.height()),
                'size': random.randint(2, 4),  # 调整粒子大小
                'speed': random.uniform(0.5, 1.5),
                'opacity': random.uniform(0.2, 0.8)
            })
        
        self.setup_ui()
        
        # 初始化计时器
        self.progress_counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(160)  # 将50ms改为160ms，这样总加载时间会增加约11秒
        
        self.particle_timer = QTimer()
        self.particle_timer.timeout.connect(self.update_particles)
        self.particle_timer.start(40)
        
        self.center_on_screen()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)  # 缩小边距
        layout.setSpacing(10)  # 缩小间距
        
        # Logo
        self.logo_label = QLabel()
        try:
            pixmap = QPixmap('logo图标.png')
            scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 缩小logo
            self.logo_label.setPixmap(scaled_pixmap)
        except:
            self.logo_label.setText("LOGO")
            self.logo_label.setStyleSheet("""
                QLabel {
                    color: #3498DB;
                    font-size: 36px;
                    font-weight: bold;
                    font-family: "Arial";
                    background-color: #ECF0F1;
                    border-radius: 40px;
                    padding: 15px;
                }
            """)
        
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        layout.addSpacing(20)
        
        # 标题
        self.title_label = QLabel('线束分析工具')
        self.title_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                font-size: 32px;
                font-weight: bold;
                font-family: "Microsoft YaHei";
                letter-spacing: 2px;
            }
        """)
        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(10)
        title_shadow.setColor(QColor(0, 0, 0, 40))
        title_shadow.setOffset(1, 1)
        self.title_label.setGraphicsEffect(title_shadow)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # 副标题
        self.subtitle_label = QLabel('专业分析·高效处理')
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #34495E;
                font-size: 18px;
                font-family: "Microsoft YaHei";
                letter-spacing: 1px;
            }
        """)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.subtitle_label)
        layout.addSpacing(40)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #F0F3F4;
                border-radius: 6px;
                text-align: center;
                border: none;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #3498DB, stop:0.5 #2E86C1, stop:1 #1A5276);
                border-radius: 6px;
            }
        """)
        
        progress_shadow = QGraphicsDropShadowEffect()
        progress_shadow.setBlurRadius(10)
        progress_shadow.setColor(QColor(0, 0, 0, 40))
        progress_shadow.setOffset(0, 2)
        self.progress_bar.setGraphicsEffect(progress_shadow)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 进度百分比
        self.percent_label = QLabel('0%')
        self.percent_label.setStyleSheet("""
            QLabel {
                color: #3498DB;
                font-size: 14px;
                font-weight: bold;
                font-family: "Arial";
            }
        """)
        self.percent_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.percent_label)
        
        # 加载提示
        self.loading_label = QLabel('正在加载系统组件...')
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 12px;
                font-family: "Microsoft YaHei";
            }
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # 版本信息
        self.version_label = QLabel('版本 1.0.0')
        self.version_label.setStyleSheet("""
            QLabel {
                color: #95A5A6;
                font-size: 11px;
                font-family: "Microsoft YaHei";
            }
        """)
        self.version_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.version_label)
        
        self.setLayout(layout)
        
        # 淡入动画
        self.setWindowOpacity(0)
        self.fade_in_animation = QPropertyAnimation(self, b'windowOpacity')
        self.fade_in_animation.setDuration(1000)
        self.fade_in_animation.setStartValue(0)
        self.fade_in_animation.setEndValue(1)
        self.fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.fade_in_animation.start()

    def center_on_screen(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def update_particles(self):
        for particle in self.particles:
            particle['y'] -= particle['speed']
            if particle['y'] < 0:
                particle['y'] = self.height()
                particle['x'] = random.randint(0, self.width())
        self.update()

    def update_progress(self):
        self.progress_counter += 1
        self.progress_bar.setValue(self.progress_counter)
        self.percent_label.setText(f"{self.progress_counter}%")
        
        loading_texts = {
            10: "正在初始化系统...",
            25: "正在加载检测模型...",
            40: "正在准备图形界面...",
            60: "正在配置算法参数...",
            75: "正在初始化神经网络...",
            90: "即将完成加载..."
        }
        
        if self.progress_counter in loading_texts:
            self.loading_label.setText(loading_texts[self.progress_counter])
        
        if self.progress_counter >= 100:
            self.fade_out_animation = QPropertyAnimation(self, b'windowOpacity')
            self.fade_out_animation.setDuration(800)
            self.fade_out_animation.setStartValue(1)
            self.fade_out_animation.setEndValue(0)
            self.fade_out_animation.setEasingCurve(QEasingCurve.InCubic)
            self.fade_out_animation.finished.connect(self.close_and_launch)
            self.fade_out_animation.start()
            
            self.timer.stop()
            self.particle_timer.stop()
    
    def close_and_launch(self):
        # 预先创建主窗口但不显示
        from mainwindows import MyApp
        self.main = MyApp()
        
        # 创建淡出动画完成后的回调函数
        def on_fade_out_finished():
            self.close()
            self.main.showMaximized()
        
        # 设置淡出动画
        self.fade_out_animation = QPropertyAnimation(self, b'windowOpacity')
        self.fade_out_animation.setDuration(800)
        self.fade_out_animation.setStartValue(1)
        self.fade_out_animation.setEndValue(0)
        self.fade_out_animation.setEasingCurve(QEasingCurve.InCubic)
        self.fade_out_animation.finished.connect(on_fade_out_finished)
        self.fade_out_animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制圆角矩形背景
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 20, 20)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 245))
        gradient.setColorAt(1, QColor(240, 240, 240, 245))
        
        painter.fillPath(path, QBrush(gradient))
        
        pen = QPen(QColor(200, 200, 200, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # 绘制粒子
        for particle in self.particles:
            painter.setOpacity(particle['opacity'])
            gradient = QRadialGradient(particle['x'], particle['y'], particle['size'])
            gradient.setColorAt(0, QColor(52, 152, 219, 200))
            gradient.setColorAt(1, QColor(52, 152, 219, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(particle['x']), int(particle['y'])), 
                              particle['size'], particle['size'])

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.log_content = "这里是日志内容"
        self.logArea = QTextEdit()

    def closeEvent(self, event):
        log_content = self.logArea.toPlainText()
        print("关闭事件被触发")
        try:
            save_log(log_content)
        except Exception as e:
            print(f"保存日志时出错: {e}")
        event.accept()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    
    # 显示启动画面
    splash = SplashScreen()
    splash.show()
    
    sys.exit(app.exec_())
