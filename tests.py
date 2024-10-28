import math
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ru_benford as bf
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import io
import sys
import io
import sys
import pandas as pd
import re
import os


class Tests:
    df = None
    OSV = None

    def __init__(self, df, OSV):
        self.df = df
        self.OSV = OSV
        self.benf = bf.Benford(df['Сумма'], decimals=2, confidence=99)
        self.first_teor = None
        self.df_duplicates = None
        self.first_real = None
        self.test_results = {}
        self.images = {}

    def test_coherence_data(self):
        df = self.df.copy().reset_index(drop=False).rename(columns={'index': 'Номер строки'})
        df['Номер строки'] = df['Номер строки'] + 2

        names = '000, 001, 002, 003, 003.01, 003.02, 004, 004.1, 004.02, 004.К, 005, 006, 007, 008, 008.1, 008.21, 009, 009.01, 009.21, 010, 011, 012, 012.01, 012.02, ГТД, КВ, МЦ, МЦ.02, МЦ.03, МЦ.04, НЕ, НЕ.01, НЕ.01.1, НЕ.01.9, НЕ.02, НЕ.02.1, НЕ.02.9, НЕ.03, НЕ.04, ОТ, ОТ.01, ОТ.02, ОТ.03, РВ, РВ.1, РВ.2, РВ.3, РВ.4, УСН, УСН.01, УСН.01, УСН.02, УСН.03, УСН.04, УСН.21, УСН.22, УСН.23, УСН.24, Я75, Я81.01, Я81, Я80, Я80.02, Я80.01, Я80.09, Я75.01, Я81.09, Я81.02, Я75.02, Я69.06.5, Я01.К, Я96, Я96.01'
        namesofaccs = [accs.strip() for accs in names.split(',')]

        # Фильтруем строки с отсутствующими или некорректными значениями в столбцах 'СчетДт' и 'СчетКт'
        result = df[df['СчетДт'].isnull() | df['СчетКт'].isnull()]
        result = result.loc[~result['СчетКт'].isin(namesofaccs) & ~result['СчетДт'].isin(namesofaccs)]

        # Проверяем результат и формируем вывод
        if result.empty:
            return pd.DataFrame([["Проверка целостности данных пройдена"]], columns=["Результат"]), True
        else:
            result = result[['Номер строки', 'СчетДт', 'СчетКт']]
            error_header = pd.DataFrame([["Есть ошибки в следующих строках:"]], columns=["Результат"])
            result_with_header = pd.concat([error_header, result], ignore_index=True)
            return result_with_header, False

    def test_math_correctly(self):
        OSV = self.OSV.copy()
        OSV['Номер строки'] = range(1, len(OSV) + 1)
        OSV['Номер строки'] = OSV['Номер строки'] + 1

        # Вычисление разницы для проверки условия и отбор строк с ошибками
        resultosv = OSV.loc[(OSV['Сальдо начальное Дт'].fillna(0) - OSV['Сальдо начальное Кт'].fillna(0) +
                             OSV['Обороты Дт'].fillna(0) - OSV['Обороты Кт'].fillna(0) -
                             OSV['Сальдо конечное Дт'].fillna(0) + OSV['Сальдо конечное Кт'].fillna(0)).abs() > 0.9]

        # Формируем результат в зависимости от наличия ошибок
        if resultosv.empty:
            return pd.DataFrame([["Проверка математической правильности успешно завершена"]],
                                columns=["Результат"]), True
        else:
            resultosv = resultosv.reset_index(drop=True)
            resultosv['Результат'] = "Ошибка в строке"
            return resultosv[['Номер строки', 'Сальдо начальное Дт', 'Сальдо начальное Кт', 'Обороты Дт',
                              'Обороты Кт', 'Сальдо конечное Дт', 'Сальдо конечное Кт', 'Результат']], False

    # Функция проверки полноты выгрузки
    def test_unloading_completeness(self):
        # Добавляем поле Номер строки в основной DataFrame и OSV
        df = self.df.copy().reset_index(drop=False).rename(columns={'index': 'Номер строки'})
        OSV = self.OSV.copy()
        OSV['Номер строки'] = range(1, len(OSV) + 1)
        OSV['Номер строки'] = OSV['Номер строки'] + 1

        journal_sum = df[['Номер строки', 'СчетДт', 'СчетКт', 'Сумма']].fillna(0)
        journal_sum_dt = journal_sum.groupby(['СчетДт'], as_index=False)['Сумма'].sum()
        journal_sum_dt['Количество Счет Дт'] = journal_sum.groupby(['СчетДт'])['СчетДт'].count()
        journal_sum_kt = journal_sum.groupby(['СчетКт'], as_index=False)['Сумма'].sum()
        journal_sum_kt['Количество Счет Кт'] = journal_sum.groupby(['СчетКт'])['СчетКт'].count()
        journal_sum_kt = journal_sum_kt.rename(columns={'Сумма': 'Обороты Кт по выгрузке'})
        journal_sum_dt = journal_sum_dt.rename(columns={'Сумма': 'Обороты Дт по выгрузке'})
        journal_sum_dt = journal_sum_dt.set_index('СчетДт')
        journal_sum_kt = journal_sum_kt.set_index('СчетКт')

        osvjournal = pd.concat([OSV, journal_sum_kt.reindex(OSV.index)], axis=1)
        osvjournal = pd.concat([osvjournal, journal_sum_dt.reindex(OSV.index)], axis=1)
        osvjournal['Обороты Дт'] = osvjournal['Обороты Дт'].fillna(0)
        osvjournal['Обороты Кт'] = osvjournal['Обороты Кт'].fillna(0)
        osvjournal['Обороты Дт по выгрузке'] = osvjournal['Обороты Дт по выгрузке'].fillna(0)
        osvjournal['Обороты Кт по выгрузке'] = osvjournal['Обороты Кт по выгрузке'].fillna(0)

        # Рассчитываем разницу по оборотам
        osvjournal['Обороты Дт разница'] = osvjournal['Обороты Дт'].round(2) - osvjournal[
            'Обороты Дт по выгрузке'].round(2)
        osvjournal['Обороты Кт разница'] = osvjournal['Обороты Кт'].round(2) - osvjournal[
            'Обороты Кт по выгрузке'].round(2)

        # Фильтруем строки с разницей
        osvresult = osvjournal.loc[(osvjournal['Обороты Дт разница'] != 0) | (osvjournal['Обороты Кт разница'] != 0)]
        osvresult = osvresult.rename(columns={'Обороты Дт': 'Обороты Дт по ОСВ', 'Обороты Кт': 'Обороты Кт по ОСВ'})

        result = osvresult[
            ["Номер строки", "Обороты Дт по ОСВ", "Обороты Кт по ОСВ", "Обороты Дт по выгрузке",
             "Обороты Кт по выгрузке", "Обороты Дт разница", "Обороты Кт разница"]]

        if len(osvresult) != 0:
            return pd.DataFrame(result), False
        else:
            return pd.DataFrame([["Проверка полноты выгрузки успешно завершена"]], columns=["Результат"]), True

    def benford_check(self):
        self.benf.summation()

        initial_size = len(self.benf.chosen)
        tested_sample_size = len(self.benf.base)
        data = (
            f"'F1D': {self.benf._discarded['F1D']}\n; "
            f"'F2D': {self.benf._discarded['F2D']}\n; "
            f"'F3D': {self.benf._discarded['F3D']}\n; "
            f"'SD': {self.benf._discarded['SD']}\n; "
            f"'L2D': {self.benf._discarded['L2D']};"
        )

        # Формируем DataFrame с нужными данными
        df = pd.DataFrame({
            'Параметр': ['Исходный размер выборки', 'Количество значений, на которых проведен тест',
                         'Исключенные значения'],
            'Значение': [initial_size, tested_sample_size, data]
        })

        return df

    def test_digits(self):
        results = {}  # Словарь для хранения всех DataFrame

        plt.ioff()

        for name, test in [('F1D', self.benf.F1D), ('SD', self.benf.SD), ('F2D', self.benf.F2D)]:
            output_buffer = io.StringIO()
            sys.stdout = output_buffer

            plot_file = f'{name}_plot.png'
            report_data = test.report(show_plot=True, save_plot=plot_file)

            sys.stdout = sys.__stdout__

            captured_output = output_buffer.getvalue()
            output_buffer.close()

            # Преобразование вывода в DataFrame
            output_df = pd.DataFrame([line] for line in captured_output.splitlines())

            # Сохраняем результат для каждого теста
            results[name] = output_df

            # Закрываем график вручную, чтобы предотвратить его вывод
            plt.close('all')

        # Возвращаем интерактивный режим, если нужно
        plt.ion()

        # Возвращаем все DataFrame
        return results

    def test_sec_order(self):
        """
        Выполняет тест второго порядка, перехватывает вывод и создает DataFrame на основе строк вывода.
        """
        # Выполнение теста второго порядка
        self.benf.sec_order()

        # Создаем строковый буфер для перехвата вывода
        report_output = io.StringIO()
        sys.stdout = report_output

        # Выполняем команду, которая выводит результаты и сохраняет график
        image_path = 'plot_F2D.png'
        self.benf.F2D_sec.report(save_plot=image_path)

        # Возвращаем стандартный вывод обратно
        sys.stdout = sys.__stdout__

        # Получаем содержимое вывода как строку
        captured_output = report_output.getvalue()

        # Создаем DataFrame на основе строк вывода
        output_df = pd.DataFrame([line] for line in captured_output.splitlines())

        # Возвращаем DataFrame с результатами и путь к изображению
        return output_df, image_path

    def test_summation(self):
        """
        Выполняет тест суммирования, перехватывает вывод и преобразует его в DataFrame.
        """
        # Перехватываем вывод команды
        report_output = io.StringIO()
        sys.stdout = report_output

        image_path = 'summation.png'

        # Выполняем тест суммирования
        sm = bf.summation(self.df['Сумма'], decimals=2, verbose=True, show_plot=True, save_plot=image_path)

        # Возвращаем стандартный вывод обратно
        sys.stdout = sys.__stdout__

        # Получаем содержимое вывода как строку
        captured_output = report_output.getvalue()

        # Преобразуем захваченный вывод в DataFrame построчно, без разделения на отдельные слова
        df_results = pd.DataFrame([line for line in captured_output.splitlines()], columns=["Отчет"])

        # Возвращаем DataFrame с результатами и путь к изображению для последующей записи в Excel
        return df_results, image_path

    def test_mantiss(self):

        # Выполняем команду, которая выводит результаты
        image_path_1 = 'plot_1.png'
        image_path_2 = 'plot_2.png'
        report_output = io.StringIO()
        sys.stdout = report_output

        # Закрываем все открытые фигуры
        plt.close('all')

        # Генерируем графики и текст
        mant = bf.mantissas(self.df['Сумма'])

        # Возвращаем стандартный вывод обратно
        sys.stdout = sys.__stdout__

        # Получаем содержимое вывода как строку
        report_text = report_output.getvalue()

        # Сохраняем все открытые графики по отдельности
        figures = [plt.figure(i) for i in plt.get_fignums()]

        # Сохраняем графики
        figures[0].savefig(image_path_1)
        figures[1].savefig(image_path_2)

        # Закрываем все открытые фигуры
        plt.close('all')

        df_results = pd.DataFrame([line.strip() for line in report_text.splitlines()], columns=["Отчет"])

        return df_results, [image_path_1, image_path_2]

    def test_amount_duplication(self):
        df = self.df

        temp = df.groupby('Сумма')['Сумма'].count() / len(df)
        sum_ = list(temp.index)
        frequency = list(temp)
        df_temp = pd.DataFrame({"Сумма": sum_, "Частота суммы": frequency})

        df = df.merge(df_temp, on='Сумма')
        most_frequent_values = df['Сумма'].value_counts().head(10)

        sorted_counts = most_frequent_values.sort_values(ascending=False)

        bar_width = 0.5  # Увеличиваем ширину столбцов
        bar_positions = [i + bar_width / 2 for i in range(len(sorted_counts))]  # Смещаем позиции столбцов

        # Построение графика
        fig, ax = plt.subplots(figsize=(10, 6))  # Используем fig и ax для сохранения фигуры
        ax.bar(bar_positions, sorted_counts.values, width=bar_width, align='center',
               color=(0 / 255, 121 / 255, 140 / 255))
        ax.set_xlabel('Значения')
        ax.set_ylabel('Частота')
        ax.set_title('Тест дублирования сумм')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(sorted_counts.index, rotation=45)

        # Добавляем проверку данных и настраиваем границы осей
        ax.set_xlim([min(bar_positions) - bar_width, max(bar_positions) + bar_width])
        ax.set_ylim([0, max(sorted_counts.values) * 1.1])  # Увеличиваем верхнюю границу на 10% для видимости

        plt.tight_layout()

        # Сохраняем график
        image_path = 'plot.png'
        fig.savefig(image_path)
        plt.close(fig)
        return pd.DataFrame(), image_path

    def test_two_last_digit(self):
        # Перехватываем вывод команды benf.F2D_sec.report()
        self.benf.sec_order()

        # Создаем строковый буфер для перехвата вывода
        report_output = io.StringIO()
        sys.stdout = report_output

        image_path = 'two_digits_plot.png'
        self.benf.L2D.report(save_plot=image_path)

        # Возвращаем стандартный вывод обратно
        sys.stdout = sys.__stdout__

        # Получаем содержимое вывода как строку
        report_text = report_output.getvalue()

        # Закрываем все открытые фигуры
        plt.close('all')

        df_results = pd.DataFrame([line.strip() for line in report_text.splitlines()], columns=["Отчет"])

        return df_results, image_path

    def calculate_coef_distortion(self):
        result = self.df.loc[self.df['Сумма'] >= 10]
        result.loc[result['Сумма'] != 0, ['Сумма']] = (
                10 * result['Сумма'] / (10 ** (np.log10(result['Сумма']).fillna(0)).astype(int)))

        Avg_Found = result['Сумма'].sum() / result['Сумма'].count()
        Avg_Expected = 90 / (result['Сумма'].count() * (10 ** (1 / result['Сумма'].count()) - 1))
        Distortion = (Avg_Found - Avg_Expected) / Avg_Expected * 100
        Std = result['Сумма'].std()
        Z_stat = Distortion / Std

        # Подготовка данных для записи в Excel
        data = {
            'Показатель': [
                'Среднее Факт', 'Среднее Теор', 'Коэффициент искажения', 'Z-статистика', 'Критическое значение',
                'Вывод по значениям', 'Значимость искажения'
            ],
            'Значение': [
                round(Avg_Found, 2), round(Avg_Expected, 2), f"{round(Distortion, 2)}%", round(Z_stat, 2), 2.57,
                'Значения в массиве занижены' if Distortion < 0 else 'Значения в массиве завышены',
                'Коэффициент искажения является существенным' if abs(
                    Z_stat) > 2.57 else 'Коэффициент искажения является несущественным'
            ]
        }

        df_output = pd.DataFrame(data)

        return df_output
