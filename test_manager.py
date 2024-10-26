import pandas as pd

from tests import Tests
from excel_writer import ExcelWriter
import matplotlib
matplotlib.use('Agg')


class TestManager:
    def __init__(self, df1, df2, output_folder, company_name, log_callback=None):
        """
        Инициализация менеджера тестов.

        :param df1: DataFrame, содержащий данные из первого файла Excel
        :param df2: DataFrame, содержащий данные из второго файла Excel
        :param output_folder: Папка для сохранения результатов
        :param company_name: Имя компании для результатов
        :param log_callback: Функция для отправки логов в UI
        """
        self.df = df1
        self.OSV = df2.round(decimals=2).set_index('Счет')
        self.tests = Tests(self.df, self.OSV)
        self.excel_writer = ExcelWriter(output_folder, company_name)
        self.log_callback = log_callback if log_callback else print  # Используем коллбек для логов или print по умолчанию

    def log(self, message):
        """Передаёт сообщение в лог."""
        self.log_callback(message)

    def run_tests(self):
        """
        Запускает все тесты и записывает результаты в Excel.
        """
        self.log("Запуск тестов согласованности данных...")
        self.excel_writer.save_data_to_excel(pd.DataFrame(), 'Тесты целостности')
        correct_data = True

        # Выполнение тестов
        consistency_result, is_success = self.tests.test_coherence_data()
        self.excel_writer.save_data_to_excel(consistency_result, 'Тест 1.1')
        if not is_success:
            correct_data = False

        self.log("Запуск тестов математической корректности...")
        math_correct, is_success = self.tests.test_math_correctly()
        self.excel_writer.save_data_to_excel(math_correct, 'Тест 1.2')
        if not is_success:
            correct_data = False

        self.log("Запуск тестов полноты выгрузки...")
        unloading_complete, is_success = self.tests.test_unloading_completeness()
        self.excel_writer.save_data_to_excel(unloading_complete, 'Тест 1.3')
        if not is_success:
            correct_data = False

        if correct_data is False:
            raise Exception('файлы с данными некорректны. Подробная информация в Excel отчете')

        self.log("Запуск теста Бенфорда...")
        benford_result = self.tests.benford_check()
        self.excel_writer.save_data_to_excel(benford_result, 'Тест Бенфорда')

        self.log("Запуск теста первой, второй и первой и второй цифры")
        digit_results = self.tests.test_digits()
        for name, df in digit_results.items():
            plot_file = f'{name}_plot.png'
            self.excel_writer.save_data_with_charts(df, f'Тест {name}', plot_file)

        self.log("Запуск теста второго порядка...")
        df_results, image_path = self.tests.test_sec_order()
        self.excel_writer.save_data_with_charts(df_results, 'Тест второго порядка', image_path)

        self.log("Запуск теста суммирования...")
        df_results, image_path = self.tests.test_summation()
        self.excel_writer.save_data_with_charts(df_results, 'Тест суммирования', image_path)

        self.log("Запуск теста мантисс...")
        df_results, image_paths = self.tests.test_mantiss()
        self.excel_writer.save_data_with_charts(df_results, 'Тест мантисс', image_paths)

        self.log("Запуск теста дублирования сумм")
        df_results, image_path = self.tests.test_amount_duplication()
        self.excel_writer.save_data_with_charts(df_results, 'Тест дублирования сумм', image_path)

        self.log("Запуск теста двух последних цифр")
        df_results, image_path = self.tests.test_two_last_digit()
        self.excel_writer.save_data_with_charts(df_results, 'Тест дублирования сумм', image_path)

        self.log("Запуск расчета коэффициента искажения")
        df_results = self.tests.calculate_coef_distortion()
        self.excel_writer.save_data_to_excel(df_results, 'Расчет коэффициента искажения')

        self.excel_writer.delete_png_files()
        self.log("Все тесты завершены.")
