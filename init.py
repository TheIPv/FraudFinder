import sys
import os
import logging
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QLineEdit, QTextEdit, QMessageBox)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFontDatabase, QFont, QDoubleValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QLocale
import pandas as pd
from test_manager import TestManager
import matplotlib
matplotlib.use('Agg')

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])


class FileLoaderThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, str)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, file_type):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):
        try:
            df = pd.read_excel(self.file_path)
            self.finished_signal.emit(df, self.file_type)
        except Exception as e:
            self.error_signal.emit(f"Ошибка при загрузке {self.file_type}: {str(e)}")


class TestRunnerThread(QThread):
    log_signal = pyqtSignal(str, str)  # Добавлен второй параметр для типа сообщения
    finished_signal = pyqtSignal()
    result_signal = pyqtSignal(object)

    def __init__(self, df1, df2, output_folder, company_name, P_A):
        super().__init__()
        self.df1 = df1
        self.df2 = df2
        self.output_folder = output_folder
        self.company_name = company_name
        self.P_A = P_A


    def run(self):
        try:
            logging.info(f"Запуск тестов для компании: {self.company_name}")
            manager = TestManager(self.df1, self.df2, self.output_folder, self.company_name, self.log, self.P_A)
            manager.run_tests()
        except Exception as e:
            self.log(f"Ошибка при выполнении тестов: {str(e)}", 'error')
        finally:
            self.finished_signal.emit()

    def log(self, message, level='info'):
        self.log_signal.emit(message, level)
        logging.info(message)


class TestApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Установка палитры для цветов фона
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('#f0f4fc'))
        palette.setColor(QPalette.Base, QColor('#ffffff'))
        self.setPalette(palette)

        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'resources', 'fraud.ico')
        else:
            icon_path = os.path.join(os.getcwd(), 'resources', 'fraud.ico')

        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle('Тестирование данных')
        self.setGeometry(100, 100, 800, 600)

        # Установка шрифта "Jury"
        font_db = QFontDatabase()
        font_id = font_db.addApplicationFont("resources\\Jura.ttf")  # Путь к файлу шрифта
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.setFont(QFont(font_family))

        layout = QVBoxLayout()

        # Заголовок
        self.label_header = QLabel('Тестирование данных')
        self.label_header.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-family: Arial, sans-serif;  /* Более современный и мягкий шрифт */
                font-weight: bold;
                color: #1976D2;  /* Синий цвет текста */
            }
        """)
        layout.addWidget(self.label_header)

        # Название компании
        self.label_company = QLabel('Название компании:')
        layout.addWidget(self.label_company)
        self.company_input = QLineEdit(self)
        self.company_input.textChanged.connect(self.check_files_and_name)
        self.style_input(self.company_input)  # Применение стиля
        layout.addWidget(self.company_input)

        # Параметр для оценки рисков
        self.label_P_A = QLabel('Вероятность мошенничества:')
        layout.addWidget(self.label_P_A)
        # Настройка текстового поля с валидатором только для значений float
        self.P_A_input = QLineEdit(self)
        float_validator = QDoubleValidator(0.00, 99.99, 2)  # Диапазон от 0.00 до 99.99 с 2 знаками после точки
        float_validator.setNotation(
            QDoubleValidator.StandardNotation)  # Только стандартный вид записи (например, без экспоненты)
        float_validator.setLocale(QLocale("en_US"))  # Устанавливаем локаль для точки вместо запятой
        self.P_A_input.setValidator(float_validator)
        # Подключение проверки при изменении текста
        self.P_A_input.textChanged.connect(self.check_files_and_name)
        self.style_input(self.P_A_input)  # Применение стиля
        layout.addWidget(self.P_A_input)

        # Кнопка загрузки файла 1
        self.load_file1_btn = QPushButton('Загрузить файл 1')
        self.load_file1_btn.clicked.connect(self.load_file_1)
        self.style_button(self.load_file1_btn)
        layout.addWidget(self.load_file1_btn)

        # Кнопка загрузки файла 2
        self.load_file2_btn = QPushButton('Загрузить файл 2')
        self.load_file2_btn.setEnabled(False)
        self.load_file2_btn.clicked.connect(self.load_file_2)
        self.style_button(self.load_file2_btn)
        layout.addWidget(self.load_file2_btn)

        # Кнопка запуска тестов
        self.run_tests_btn = QPushButton('Запустить тесты')
        self.run_tests_btn.setEnabled(False)
        self.run_tests_btn.clicked.connect(self.run_tests)
        self.style_button(self.run_tests_btn)
        layout.addWidget(self.run_tests_btn)

        # Лог вывода
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        # Переменные для файлов
        self.df1 = None
        self.df2 = None

    def style_button(self, button):
        """Устанавливаем стиль для кнопок."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #E1ECF4;  /* Светло-голубой для фона кнопок */
                color: #333333;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                outline: none;
            }
            QPushButton:hover {
                background-color: #C6D9EE;  /* Легкий эффект при наведении */
            }
            QPushButton:disabled {
                background-color: #DADADA;  /* Бледный серый, когда кнопка отключена */
                color: #A0A0A0;
            }
        """)

    def style_input(self, input_field):
        """Применение стиля для поля ввода."""
        input_field.setStyleSheet("""
            QLineEdit {
                background-color: #F5F7FA;
                border: 1px solid #D3D3D3;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
            }
        """)

    def update_log(self, message, level='info'):
        """Обновление логов с цветовым кодированием и форматированием."""
        if level == 'info':
            self.log_output.append(f"<span style='color: Z#0d0d0d;'>{message}</span>")  # Синий для обычной информации
        elif level == 'warning':
            self.log_output.append(f"<span style='color: #FF9800; font-weight: bold;'>{message}</span>")  # Оранжевый для предупреждений
        elif level == 'error':
            self.log_output.append(f"<span style='color: #D32F2F; font-weight: bold;'>{message}</span>")  # Красный для ошибок
        else:
            self.log_output.append(message)  # По умолчанию — обычный текст

    def load_file_1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл 1", "", "Excel Files (*.xlsx);;All Files (*)")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_path:
            self.update_log(f"Загрузка {file_name}...", 'info')
            self.thread1 = FileLoaderThread(file_path, "файл 1")
            self.thread1.finished_signal.connect(self.on_file_loaded)
            self.thread1.error_signal.connect(self.on_file_error)
            self.thread1.start()

    def load_file_2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл 2", "", "Excel Files (*.xlsx);;All Files (*)")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_path:
            self.update_log(f"Загрузка файла {file_name}...", 'info')
            self.thread2 = FileLoaderThread(file_path, "файл 2")
            self.thread2.finished_signal.connect(self.on_file_loaded)
            self.thread2.error_signal.connect(self.on_file_error)
            self.thread2.start()

    def on_file_loaded(self, df, file_type):
        if file_type == "файл 1":
            self.df1 = df
            self.load_file2_btn.setEnabled(True)
            self.update_log(f"Файл успешно загружен.", 'info')
        elif file_type == "файл 2":
            self.df2 = df
            self.update_log(f"Файл успешно загружен.", 'info')
        self.check_files_and_name()

    def on_file_error(self, message):
        self.update_log(message, 'error')
        logging.error(message)

    def check_files_and_name(self):
        if self.df1 is not None and self.df2 is not None and self.company_input.text().strip():
            self.run_tests_btn.setEnabled(True)
        else:
            self.run_tests_btn.setEnabled(False)

    def run_tests(self):

        text = self.P_A_input.text()
        if ',' in text:
            self.P_A_input.setText(text.replace(',', '.'))
        P_A = float(self.P_A_input.text())

        if not P_A:
            self.update_log('Пожалуйста, введите показатель мошенничества.', 'warning')
            return

        company_name = self.company_input.text()
        if not company_name:
            self.update_log('Пожалуйста, введите название компании.', 'warning')
            return

        output_folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if not output_folder:
            self.update_log('Пожалуйста, выберите папку для сохранения.', 'warning')
            return

        self.update_log('Запуск тестов...', 'info')
        self.thread = TestRunnerThread(self.df1, self.df2, output_folder, company_name, P_A)
        self.thread.log_signal.connect(self.update_log)
        self.thread.finished_signal.connect(self.on_tests_finished)
        self.thread.start()

    def on_tests_finished(self):
        self.update_log("Тесты завершены.", 'info')

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Завершено")
        msg_box.setText("Отчет сформирован успешно.")

        # Установка стиля для уменьшения шрифта текста и размеров кнопок
        msg_box.setStyleSheet("""
            QMessageBox {
                font-size: 15px;  /* Уменьшение размера текста */
            }
            QPushButton {
                font-size: 15px;  /* Уменьшение размера текста на кнопке */
            }
        """)

        msg_box.exec_()


app = QApplication(sys.argv)

font_db = QFontDatabase()
font_id = font_db.addApplicationFont("resources\\Jura.ttf")
if font_id != -1:
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
    app.setFont(QFont(font_family, 10))

ex = TestApp()
ex.show()
sys.exit(app.exec_())