from test_manager import TestManager
import pandas as pd
import logging

file_path_1 = "resources/ЖО Дубнадорстрой.xlsx"
file_path_2 = "resources/ОСВ Дубнадорстрой.xlsx"
output_folder = "C:\\Users\\Adam\\Documents\\PyCharmProjects\\pythonProject4\\es-project"
excel_filename = "Результаты тестов"

df = pd.read_excel(file_path_1, converters={'Счет Кт': str, 'Счет Дт': str})
OSV = pd.read_excel(file_path_2)

def log(message):
    logging.info(message)  # Отправка логов в консоль


manager = TestManager(df, OSV, output_folder, excel_filename, log)
manager.run_tests()
