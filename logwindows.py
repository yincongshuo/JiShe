import sys
import os
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLineEdit, QTextEdit, QPushButton, QApplication
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRegularExpression
from PyQt5.QtGui import QTextCharFormat, QTextCursor
from datetime import datetime
import chardet

class BaseLog(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('logwindows.ui', self)
        self.init_image_viewer_and_controls()
        self.init_signal_slots()
        self.init_animations()
        self.init_log_styles()

    def init_log_styles(self):
        """初始化日志样式"""
        self.log_formats = {
            'INFO': QTextCharFormat(),
            'WARNING': QTextCharFormat(),
            'ERROR': QTextCharFormat()
        }
        
        # 设置中文字体
        font = QtGui.QFont()
        # 设置多个备选字体，确保至少一个可用
        font.setFamilies(['Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC'])
        font.setPointSize(10)  # 设置合适的字体大小
        
        # 普通日志样式 - 黑色
        self.log_formats['INFO'].setForeground(QtGui.QColor('#2C3E50'))
        self.log_formats['INFO'].setFont(font)
        
        # 警告日志样式 - 橙色
        self.log_formats['WARNING'].setForeground(QtGui.QColor('#F39C12'))
        self.log_formats['WARNING'].setFont(font)
        self.log_formats['WARNING'].setFontWeight(QtGui.QFont.Bold)
        
        # 错误日志样式 - 红色
        self.log_formats['ERROR'].setForeground(QtGui.QColor('#E74C3C'))
        self.log_formats['ERROR'].setFont(font)
        self.log_formats['ERROR'].setFontWeight(QtGui.QFont.Bold)

    def format_log_text(self, text):
        """根据日志类型设置不同的颜色"""
        cursor = self.logArea.textCursor()
        lines = text.split('\n')
        
        for line in lines:
            if line.strip():
                if '[ERROR]' in line:
                    cursor.insertText(line + '\n', self.log_formats['ERROR'])
                elif '[WARNING]' in line:
                    cursor.insertText(line + '\n', self.log_formats['WARNING'])
                else:
                    cursor.insertText(line + '\n', self.log_formats['INFO'])

    def show_log(self):
        """显示日志内容"""
        filenames = 'log.txt'
        try:
            if not os.path.exists(filenames):
                self.logArea.clear()
                self.format_log_text("[WARNING] 日志文件未找到。")
                self.show()
                return

            # 读取文件内容
            try:
                # 首先尝试使用UTF-8-SIG（带BOM）读取
                with open(filenames, 'r', encoding='utf-8-sig') as f:
                    log_data = f.read()
            except UnicodeError:
                try:
                    # 如果失败，尝试使用普通UTF-8读取
                    with open(filenames, 'r', encoding='utf-8') as f:
                        log_data = f.read()
                except UnicodeError:
                    try:
                        # 如果还是失败，尝试使用系统默认编码
                        with open(filenames, 'r') as f:
                            log_data = f.read()
                    except UnicodeError:
                        # 如果所有尝试都失败，使用二进制模式读取并尝试解码
                        with open(filenames, 'rb') as f:
                            raw_data = f.read()
                            # 使用chardet检测编码
                            result = chardet.detect(raw_data)
                            if result['encoding'] and result['confidence'] > 0.5:
                                try:
                                    log_data = raw_data.decode(result['encoding'])
                                except:
                                    # 如果还是失败，尝试忽略错误字符
                                    log_data = raw_data.decode('utf-8', errors='ignore')
                            else:
                                # 如果检测失败，使用UTF-8并忽略错误
                                log_data = raw_data.decode('utf-8', errors='ignore')
            
            self.logArea.clear()  # 清空现有内容
            self.format_log_text(log_data)
            
            # 将光标移动到文档末尾
            cursor = self.logArea.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.logArea.setTextCursor(cursor)
            
            # 确保滚动到最底部
            self.logArea.verticalScrollBar().setValue(
                self.logArea.verticalScrollBar().maximum()
            )
            
            self.show()
            
        except Exception as e:
            error_msg = f"[ERROR] 读取日志文件时发生错误: {str(e)}"
            print(error_msg)
            self.logArea.clear()
            self.format_log_text(error_msg)
            self.show()

    def init_image_viewer_and_controls(self):
        """初始化控件"""
        self.searchBar = self.findChild(QLineEdit, "searchBar")
        self.logArea = self.findChild(QTextEdit, "logArea")
        self.saveButton = self.findChild(QPushButton, "saveButton")
        self.exportButton = self.findChild(QPushButton, "exportButton")


    def init_signal_slots(self):
        """初始化信号槽连接"""
        if self.searchBar:
            self.searchBar.textChanged.connect(self.searchLog)
        if self.saveButton:
            self.saveButton.clicked.connect(lambda: self.animate_button(self.saveButton))
            self.saveButton.clicked.connect(self.saveLog)
        if self.exportButton:
            self.exportButton.clicked.connect(lambda: self.animate_button(self.exportButton, self.exportLog))

    def init_animations(self):
        """初始化动画设置"""
        self.animations = {}

    def animate_button(self, button, callback=None):
        """按钮点击时的缩放动画"""
        if button in self.animations:
            self.animations[button].stop()

        enlarge_animation = QPropertyAnimation(button, b"size")
        enlarge_animation.setDuration(50)
        enlarge_animation.setStartValue(button.size())
        enlarge_animation.setEndValue(button.size() + QtCore.QSize(3, 3))
        enlarge_animation.setEasingCurve(QEasingCurve.OutBounce)

        shrink_animation = QPropertyAnimation(button, b"size")
        shrink_animation.setDuration(50)
        shrink_animation.setStartValue(button.size() + QtCore.QSize(3, 3))
        shrink_animation.setEndValue(button.size())
        shrink_animation.setEasingCurve(QEasingCurve.InBounce)

        enlarge_animation.finished.connect(lambda: shrink_animation.start())

        if callback:
            shrink_animation.finished.connect(callback)

        self.animations[button] = enlarge_animation
        enlarge_animation.start()

    def searchLog(self, text):
        """在日志区域中搜索并高亮显示文本"""
        document = self.logArea.document()
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(Qt.yellow)
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())

        if text:
            regex = QRegularExpression(text)
            cursor = QTextCursor(document)
            while not cursor.isNull() and not cursor.atEnd():
                cursor = document.find(regex, cursor)
                if not cursor.isNull():
                    cursor.mergeCharFormat(highlight_format)

    def saveLog(self):
        """保存当前日志内容到文件"""
        try:
            # 保存时统一使用UTF-8-SIG编码（带BOM）
            with open('log.txt', 'w', encoding='utf-8-sig') as f:
                log_data = self.logArea.toPlainText()
                f.write(log_data)
        except Exception as e:
            error_msg = f"[ERROR] 保存日志时发生错误: {str(e)}"
            print(error_msg)
            self.format_log_text(error_msg)

    def exportLog(self):
        """导出日志内容到指定文件"""
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "导出日志", "", "Text Files (*.txt);;All Files (*)",
                                                  options=options)
        if filename:
            try:
                # 导出时也使用UTF-8-SIG编码（带BOM）
                with open(filename, 'w', encoding='utf-8-sig') as f:
                    log_data = self.logArea.toPlainText()
                    f.write(log_data)
            except Exception as e:
                error_msg = f"[ERROR] 导出日志时发生错误: {str(e)}"
                print(error_msg)
                self.format_log_text(error_msg)

    def add_log(self, message, level='INFO'):
        """添加日志，可以指定日志级别"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{level}] [{timestamp}] {message}"
        cursor = self.logArea.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.logArea.setTextCursor(cursor)
        self.format_log_text(log_message)

class ErrorLog(BaseLog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("错误日志")
        self.logArea.setStyleSheet("""
            QTextEdit {
                font-family: "Microsoft YaHei", "SimHei", "SimSun";
                font-size: 12pt;
                border: 1px solid #E74C3C;
                border-radius: 18px;
                background-color: #FFFFFF;
                padding: 12px;
            }
        """)

class WarningLog(BaseLog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("警告日志")
        self.logArea.setStyleSheet("""
            QTextEdit {
                font-family: "Microsoft YaHei", "SimHei", "SimSun";
                font-size: 12pt;
                border: 1px solid #F39C12;
                border-radius: 18px;
                background-color: #FFFFFF;
                padding: 12px;
            }
        """)

class InfoLog(BaseLog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信息日志")
        self.logArea.setStyleSheet("""
            QTextEdit {
                font-family: "Microsoft YaHei", "SimHei", "SimSun";
                font-size: 12pt;
                border: 1px solid #2C3E50;
                border-radius: 18px;
                background-color: #FFFFFF;
                padding: 12px;
            }
        """)

def save_log(log_content):
    """保存日志到文件"""
    try:
        with open('log.txt', 'w', encoding='gbk') as f:
            f.write(log_content)
    except Exception as e:
        print(f"[ERROR] 保存日志时出错: {e}")




