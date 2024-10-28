import os
import datetime
import tkinter as tk
import threading
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Отключает отображение графиков на экране

# Глобальные переменные для хранения загруженных данных
df = None
OSV = None


# Функция для создания уникальной папки на рабочем столе
def create_output_folder():
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(desktop, f'Анализ_{timestamp}')
    os.makedirs(folder_path)
    return folder_path


# Функция для обновления лога
def update_log(message):
    log_label.config(text=message)
    window.update_idletasks()


# Функция для загрузки файла
def load_file(file_type):
    update_log(f"Файл {file_type} загружается...")
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if not file_path:
        messagebox.showwarning("Ошибка", f"Файл {file_type} не выбран!")
        return None

    if file_type == "ЖО":
        global df
        df = pd.read_excel(file_path, converters={'Счет Кт': str, 'Счет Дт': str})
    else:  # file_type == "ОСВ"
        global OSV
        OSV = pd.read_excel(file_path).round(decimals=2).set_index('Счет')

    update_log(f"Файл {file_type} загружен")
    return file_path


# Функция для загрузки первого файла
def load_first_file():
    load_file("ЖО")
    load_osv_button.config(state=tk.NORMAL)


# Функция для загрузки второго файла
def load_second_file():
    global df, OSV
    load_file("ОСВ")
    if df is not None and OSV is not None:
        update_log("Файлы загружены. Готово к анализу.")
        start_analysis_button.pack(pady=10)


def analyze_data_thread():
    analyze_data()


def start_analysis():
    threading.Thread(target=analyze_data_thread).start()


# Функция для анализа данных
def analyze_data():
    global df, OSV
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tkinter as tk
    import os
    from tkinter import filedialog, messagebox
    update_log("Тест на целостность данных...")
    window.update_idletasks()  # Обновляем интерфейс

    # Тест целостности данных
    names = '000, 001, 002, 003, 003.01, 003.02, 004, 004.1, 004.02, 004.К, 005, 006, 007, 008, 008.1, 008.21, 009, 009.01, 009.21, 010, 011, 012, 012.01, 012.02, ГТД, КВ, МЦ, МЦ.02, МЦ.03, МЦ.04, НЕ, НЕ.01, НЕ.01.1, НЕ.01.9, НЕ.02, НЕ.02.1, НЕ.02.9, НЕ.03, НЕ.04, ОТ, ОТ.01, ОТ.02, ОТ.03, РВ, РВ.1, РВ.2, РВ.3, РВ.4, УСН, УСН.01, УСН.01, УСН.02, УСН.03, УСН.04, УСН.21, УСН.22, УСН.23, УСН.24, Я75, Я81.01, Я81, Я80, Я80.02, Я80.01, Я80.09, Я75.01, Я81.09, Я81.02, Я75.02, Я69.06.5, Я01.К, Я96, Я96.01'
    namesofaccs = [accs.strip() for accs in names.split(',')]
    result = df[df['СчетДт'].isnull() | df['СчетКт'].isnull()]
    result = result.loc[~result['СчетКт'].isin(namesofaccs) & ~result['СчетДт'].isin(namesofaccs)]

    if result.shape[0] == 0:
        print("Проверка целостности данных успешно завершена")
    else:
        print("Проверка целостности данных выполнена с ошибками")
    update_log("Тест математической правильности...")
    # Тест математической правильности
    resultosv = OSV.loc[(OSV['Сальдо начальное Дт'].fillna(0) - OSV['Сальдо начальное Кт'].fillna(0) +
                         OSV['Обороты Дт'].fillna(0) - OSV['Обороты Кт'].fillna(0) -
                         OSV['Сальдо конечное Дт'].fillna(0) + OSV['Сальдо конечное Кт'].fillna(0)).abs() > 0.9]

    if resultosv.shape[0] == 0:
        print("Проверка математической правильности успешно завершена")
    else:
        print("Проверка математической правильности выполнена с ошибками")
    update_log("Тест полноты выгрузки...")
    # Тест полноты выгрузки
    journal_sum = df[['СчетДт', 'СчетКт', 'Сумма']].fillna(0)
    journal_sum_dt = journal_sum.groupby(['СчетДт'], as_index=False)['Сумма'].sum()
    journal_sum_kt = journal_sum.groupby(['СчетКт'], as_index=False)['Сумма'].sum()
    journal_sum_kt = journal_sum_kt.rename(columns={'Сумма': 'Обороты Кт по выгрузке'})
    journal_sum_dt = journal_sum_dt.rename(columns={'Сумма': 'Обороты Дт по выгрузке'}).set_index('СчетДт')
    journal_sum_kt = journal_sum_kt.set_index('СчетКт')
    osvjournal = pd.concat([OSV, journal_sum_kt.reindex(OSV.index)], axis=1)
    osvjournal = pd.concat([osvjournal, journal_sum_dt.reindex(OSV.index)], axis=1)
    osvjournal['Обороты Дт разница'] = osvjournal['Обороты Дт'].round(2) - osvjournal['Обороты Дт по выгрузке'].round(2)
    osvjournal['Обороты Кт разница'] = osvjournal['Обороты Кт'].round(2) - osvjournal['Обороты Кт по выгрузке'].round(2)
    osvresult = osvjournal.loc[(osvjournal['Обороты Дт разница'] != 0) | (osvjournal['Обороты Кт разница'] != 0)]

    if len(osvresult) != 0:
        print("Проверка полноты выгрузки выполнена с ошибками")
    else:
        print("Проверка полноты выгрузки успешно завершена")

    """# **Проверка законом Бенфрода**

                ## **Вспомогательный код**
                """
    update_log("Проведение вспомогательных вычислений...")
    import ru_benford as bf
    benf = bf.Benford(df['Сумма'], decimals=2, confidence=99)

    benf.summation()

    # Добавляем признак: является ли сумма сторнированной (1) или нет (0)
    df["Сторно"] = df["Сумма"].apply(lambda x: 0 if x > 0 else 1)

    # Убираем строки с пустыми значениями поля Сумма
    df = df[df['Сумма'].notnull()]

    # Для отрицательных сумм меняем знак
    df.loc[df["Сумма"] < 0, "Сумма"] = -df["Сумма"]

    # Оставляем значения, больше 10, чтобы на них можно было провести все тесты
    df = df[df['Сумма'] > 10]

    # Меняем индексы
    df = df.reset_index(drop=True)
    """## **2.1 Базовые тесты**

    ### **2.1.1 Тест первой цифры**
    """

    benf.F1D.report()

    """### **2.1.2 Тест второй цифры**"""

    benf.SD.report()

    """### **2.1.3 Тест первых двух цифр**"""

    benf.F2D.report()

    """## **2.2 Продвинутые тесты**

    ### **2.2.1 Тест суммирования**
    """

    sm = bf.summation(df['Сумма'], decimals=2)

    """### **2.2.2 Тест второго порядка**"""

    benf.sec_order()
    benf.F2D_sec.report()

    """### **2.2.3 Тест мантисс**"""

    mant = bf.mantissas(df['Сумма'])

    """## **2.3 Связанные тесты**

    ### **2.3.1 Тест дублирования сумм**
    """

    # Считаем частоту появления сумм

    temp = df.groupby('Сумма')['Сумма'].count() / len(df)
    sum_ = list(temp.index)
    frequency = list(temp)
    df_temp = pd.DataFrame({"Сумма": sum_, "Частота суммы": frequency})

    df = df.merge(df_temp, on='Сумма')
    test_overwrite = df
    most_frequent_values = df['Сумма'].value_counts().head(10)

    sorted_counts = most_frequent_values.sort_values(ascending=False)

    bar_width = 0.5  # Увеличиваем ширину столбцов
    bar_positions = [i + bar_width / 2 for i in range(len(sorted_counts))]  # Смещаем позиции столбцов

    plt.figure(figsize=(10, 6))  # Увеличиваем размер графика
    plt.bar(bar_positions, sorted_counts.values, width=bar_width, align='center',
            color=(0 / 255, 121 / 255, 140 / 255))  # Устанавливаем цвет в RGB
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.title('Тест дублирования сумм')
    plt.xticks(bar_positions, sorted_counts.index, rotation=45)  # Вращаем подписи для лучшей читаемости
    plt.tight_layout()  # Оптимально размещаем элементы на графике
    # plt.show()

    """### **2.3.2 Тест двух последних цифр**"""

    benf.L2D.report()

    """### **2.3.3 Оценка коэффициента искажения**"""

    result = df.loc[df['Сумма'] >= 10]
    result.loc[result['Сумма'] != 0, ['Сумма']] = (
            10 * result['Сумма'] / (10 ** (np.log10(result['Сумма']).fillna(0)).astype(int)))

    Avg_Found = result['Сумма'].sum() / result['Сумма'].count()
    Avg_Expected = 90 / (result['Сумма'].count() * (10 ** (1 / result['Сумма'].count()) - 1))
    Distortion = (Avg_Found - Avg_Expected) / Avg_Expected * 100
    Std = result['Сумма'].std()
    Z_stat = Distortion / Std

    print(f'Среднее Факт: {round(Avg_Found, 2)}')
    print(f'Среднее Теор: {round(Avg_Expected, 2)}')
    print(f'Коэффициент искажения: {round(Distortion, 2)}%')
    print(f'Z-статистика: {round(Z_stat, 2)}')
    print('Критическое значение: 2,57')

    if (Distortion < 0):
        print('Вывод: Значения в массиве занижены')
    else:
        print('Вывод: Значения в массиве завышены')
    if (abs(Z_stat) > 2, 57):
        print('Коэффициент искажения является существенным')
    else:
        print('Коэффициент искажения является несущественным')

    """# **Оценка риска**

    ## **2. Закон Бенфорда**
    """
    update_log("Выгрузка законов Бенфорта...")
    benf.summation()

    initial_size = len(benf.chosen)
    tested_sample_size = len(benf.base)
    data = (
        f"'F1D': {benf._discarded['F1D']}\n; "
        f"'F2D': {benf._discarded['F2D']}\n; "
        f"'F3D': {benf._discarded['F3D']}\n; "
        f"'SD': {benf._discarded['SD']}\n; "
        f"'L2D': {benf._discarded['L2D']};"
    )
    # Путь к папке для сохранения отчетов
    output_folder = create_output_folder()
    # Путь для сохранения Excel-файла
    output_path = os.path.join(output_folder, 'Тесты Бенфорта (F1D, F2D, F3D, SD, L2D).xlsx')

    # Запись данных в Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Создание листа "Результаты тестов Бенфорда"
        worksheet = writer.book.add_worksheet('Результаты тестов')

        # Запись описания и значений
        worksheet.write('A1', 'Исходный размер выборки')
        worksheet.write('B1', initial_size)

        worksheet.write('A2', 'Количество значений, на которых проведен тест')
        worksheet.write('B2', tested_sample_size)

        worksheet.write('A3', 'Исключенные значения для каждого теста')
        worksheet.write('B3', data)

    # Файл будет сохранен в вашем текущем каталоге
    print(f'Отчет сохранен в файле: {output_path}')
    update_log("Выгрузка тестов первой, второй и 1 и 2 цифры...")
    """## **3. Тест первой, второй и 1и2 цифры**"""

    import pandas as pd
    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image
    import io
    import sys

    file_name = os.path.join(output_folder, 'Тесты Бенфорта, графики (F1D, SD, F2D).xlsx')

    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for name, test in [('F1D', benf.F1D), ('SD', benf.SD), ('F2D', benf.F2D)]:
            output_buffer = io.StringIO()

            sys.stdout = output_buffer

            report_data = test.report(show_plot=False)

            sys.stdout = sys.__stdout__

            captured_output = output_buffer.getvalue()
            output_buffer.close()

            # Преобразование вывода в DataFrame
            output_df = pd.DataFrame([line] for line in captured_output.splitlines())
            output_df.to_excel(writer, sheet_name=f'{name} Report', header=False, index=False, startrow=0)

            if report_data is not None:
                if isinstance(report_data, pd.DataFrame):
                    report_data.to_excel(writer, sheet_name=f'{name} Report', index=True,
                                         startrow=len(output_df) + 2)

    # Открытие книги Excel для добавления изображений
    workbook = load_workbook(file_name)

    for name in ['F1D', 'SD', 'F2D']:
        plot_file = f'{name}_plot.png'
        test = getattr(benf, name)
        test.report(show_plot=True, save_plot=plot_file)  # Сохранение графика в файл

        worksheet = workbook[f'{name} Report']
        img = Image(plot_file)

        # Настройка размера изображения (уменьшение до 50% от оригинального)
        img.width = img.width // 2
        img.height = img.height // 2

        # Определение позиции изображения (правее таблицы)
        max_column = worksheet.max_column  # Находим последнюю колонку с данными
        image_column = max_column + 2  # Смещаемся на 2 колонки вправо

        # Добавляем изображение в нужную позицию
        worksheet.add_image(img, f'{chr(65 + image_column)}5')  # Столбец A + image_column

    # Сохранение изменений в файле
    workbook.save(file_name)

    print(f"Отчеты, таблицы и графики успешно сохранены в файл {file_name}")

    """## **4. Тест суммирования**"""

    """## **5. Тест второго порядка**"""
    update_log("Выгрузка теста второго порядка...")
    import io
    import sys
    import pandas as pd
    import re

    # Перехватываем вывод команды benf.F2D_sec.report()
    benf.sec_order()

    # Создаем строковый буфер для перехвата вывода
    report_output = io.StringIO()
    sys.stdout = report_output

    # Выполняем команду, которая выводит результаты
    image_path = 'plot.png'
    benf.F2D_sec.report(save_plot=image_path)

    # Возвращаем стандартный вывод обратно
    sys.stdout = sys.__stdout__

    # Получаем содержимое вывода как строку
    report_text = report_output.getvalue()

    # Парсим данные с помощью регулярных выражений для создания таблицы
    pattern = re.compile(r'(\d{1,2})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)')
    parsed_data = pattern.findall(report_text)

    # Создаем списки для DataFrame с результатами
    first_two_digits = []
    expected = []
    found = []
    z_scores = []

    for entry in parsed_data:
        first_two_digits.append(entry[0])
        expected.append(float(entry[1]))
        found.append(float(entry[2]))
        z_scores.append(float(entry[3]))

    # Создаем DataFrame с четкими заголовками
    df_results = pd.DataFrame({
        'Первые цифры': first_two_digits,
        'Теор': expected,
        'Факт': found,
        'Z_статистика': z_scores
    })

    # Удаляем строки таблицы из полного текста отчета и дублирующиеся заголовки
    report_lines = [line.strip() for line in report_text.splitlines() if not pattern.match(line)]
    report_lines_cleaned = []
    skip_next = False

    for line in report_lines:
        if "Теор" in line and "Факт" in line and "Z_статистика" in line:
            if not skip_next:
                skip_next = True
                continue
        if skip_next:
            skip_next = False
            continue
        report_lines_cleaned.append(line)

    # Записываем данные в Excel
    with pd.ExcelWriter(os.path.join(output_folder, 'Тест второго порядка.xlsx'), engine='xlsxwriter') as writer:
        # Создаем лист для результатов анализа
        worksheet_results = writer.book.add_worksheet('тест второго порядка')

        # Записываем очищенный полный отчет, разбивая строки по пробелам
        for row_num, line in enumerate(report_lines_cleaned):
            columns = line.split()
            for col_num, cell_value in enumerate(columns):
                worksheet_results.write(row_num, col_num, cell_value)

        # Определяем начальную строку для вставки DataFrame
        start_row = len(report_lines_cleaned) + 2  # Добавляем небольшой отступ

        # Записываем DataFrame вниз под текстовым отчетом
        for col_num, header in enumerate(df_results.columns):
            worksheet_results.write(start_row, col_num, header)  # Записываем заголовки

        for row_num, (digit, exp, found_val, z) in enumerate(
                zip(df_results['Первые цифры'], df_results['Теор'], df_results['Факт'],
                    df_results['Z_статистика']),
                start=start_row + 1):
            worksheet_results.write(row_num, 0, digit)  # First_2_Dig
            worksheet_results.write(row_num, 1, exp)  # Expected
            worksheet_results.write(row_num, 2, found_val)  # Found
            worksheet_results.write(row_num, 3, z)  # Z_score

        # Растягиваем ширину колонок
        for col_num, _ in enumerate(df_results.columns):
            max_width = max(
                [len(str(value)) for value in df_results.iloc[:, col_num]] + [len(df_results.columns[col_num])])
            worksheet_results.set_column(col_num, col_num, max_width + 2)  # Добавляем небольшой отступ

        # Вставляем график в Excel со столбца K
        worksheet_results.insert_image('K1', image_path, {'x_scale': 0.5, 'y_scale': 0.5})

    # Удаляем временный файл изображения
    import os
    os.remove(image_path)
    update_log("Выгрузка теста мантисс...")
    """## **6. Тест мантисс**"""

    import io
    import sys
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    # Выполняем команду, которая выводит результаты
    image_path_1 = 'plot_1.png'
    image_path_2 = 'plot_2.png'
    report_output = io.StringIO()
    sys.stdout = report_output

    # Закрываем все открытые фигуры
    plt.close('all')

    # Генерируем графики и текст
    mant = bf.mantissas(df['Сумма'])

    # Возвращаем стандартный вывод обратно
    sys.stdout = sys.__stdout__

    # Получаем содержимое вывода как строку
    report_text = report_output.getvalue()
    report_lines = [line.strip() for line in report_text.splitlines()]

    # Сохраняем все открытые графики по отдельности
    figures = [plt.figure(i) for i in plt.get_fignums()]

    # Сохраняем графики
    figures[0].savefig(image_path_1)
    figures[1].savefig(image_path_2)

    # Закрываем все открытые фигуры
    plt.close('all')

    # Записываем данные в Excel
    with pd.ExcelWriter(os.path.join(output_folder, 'Тест Мантисс.xlsx'), engine='xlsxwriter') as writer:
        # Создаем лист для результатов анализа
        worksheet_results = writer.book.add_worksheet('Тест Мантисс')

        # Записываем очищенный полный отчет, разбивая строки по пробелам
        for row_num, line in enumerate(report_lines):
            columns = line.split()
            for col_num, cell_value in enumerate(columns):
                worksheet_results.write(row_num, col_num, cell_value)

        # Вставляем графики в Excel
        worksheet_results.insert_image('A12', image_path_1, {'x_scale': 0.5, 'y_scale': 0.5})
        worksheet_results.insert_image('K1', image_path_2, {'x_scale': 0.5, 'y_scale': 0.5})

    # Удаляем временные файлы изображений, если они существуют
    os.remove(image_path_1)
    os.remove(image_path_2)

    """## **7. Связанные тесты**"""
    update_log("Выгрузка связанных тестов...")
    "# 7.1 Тест дублирования cумм"
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
    fig.savefig(image_path)  # Сохраняем график через fig.savefig
    plt.close(fig)  # Закрываем конкретную фигуру

    # Запись в Excel
    with pd.ExcelWriter(os.path.join(output_folder, 'Тест дублирования сумм.xlsx'), engine='xlsxwriter') as writer:
        worksheet_results = writer.book.add_worksheet('Тест дублирования сумм')
        worksheet_results.insert_image('A1', image_path)

    # Удаляем временный файл
    os.remove(image_path)

    "# 7.2 Тест двух последних цифр"

    import io
    import sys
    import pandas as pd
    import re

    # Перехватываем вывод команды benf.F2D_sec.report()
    benf.sec_order()

    # Создаем строковый буфер для перехвата вывода
    report_output = io.StringIO()
    sys.stdout = report_output

    # Выполняем команду, которая выводит результаты
    image_path = 'plot.png'
    benf.L2D.report(save_plot=image_path)

    # Возвращаем стандартный вывод обратно
    sys.stdout = sys.__stdout__

    # Получаем содержимое вывода как строку
    report_text = report_output.getvalue()

    # Парсим данные с помощью регулярных выражений для создания таблицы
    pattern = re.compile(r'(\d{1,2})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)')
    parsed_data = pattern.findall(report_text)

    # Создаем списки для DataFrame с результатами
    first_two_digits = []
    expected = []
    found = []
    z_scores = []

    for entry in parsed_data:
        first_two_digits.append(entry[0])
        expected.append(float(entry[1]))
        found.append(float(entry[2]))
        z_scores.append(float(entry[3]))

    # Создаем DataFrame с четкими заголовками
    df_results = pd.DataFrame({
        'Первые цифры': first_two_digits,
        'Теор': expected,
        'Факт': found,
        'Z_статистика': z_scores
    })

    # Удаляем строки таблицы из полного текста отчета и дублирующиеся заголовки
    report_lines = [line.strip() for line in report_text.splitlines() if not pattern.match(line)]
    report_lines_cleaned = []
    skip_next = False

    for line in report_lines:
        if "Теор" in line and "Факт" in line and "Z_статистика" in line:
            if not skip_next:
                skip_next = True
                continue
        if skip_next:
            skip_next = False
            continue
        report_lines_cleaned.append(line)

    # Записываем данные в Excel
    with pd.ExcelWriter(os.path.join(output_folder, 'Тест двух последних цифр.xlsx'), engine='xlsxwriter') as writer:
        # Создаем лист для результатов анализа
        worksheet_results = writer.book.add_worksheet('тест двух последних цифр')

        # Записываем очищенный полный отчет, разбивая строки по пробелам
        for row_num, line in enumerate(report_lines_cleaned):
            columns = line.split()
            for col_num, cell_value in enumerate(columns):
                worksheet_results.write(row_num, col_num, cell_value)

        # Определяем начальную строку для вставки DataFrame
        start_row = len(report_lines_cleaned) + 2  # Добавляем небольшой отступ

        # Записываем DataFrame вниз под текстовым отчетом
        for col_num, header in enumerate(df_results.columns):
            worksheet_results.write(start_row, col_num, header)  # Записываем заголовки

        for row_num, (digit, exp, found_val, z) in enumerate(
                zip(df_results['Первые цифры'], df_results['Теор'], df_results['Факт'],
                    df_results['Z_статистика']),
                start=start_row + 1):
            worksheet_results.write(row_num, 0, digit)  # First_2_Dig
            worksheet_results.write(row_num, 1, exp)  # Expected
            worksheet_results.write(row_num, 2, found_val)  # Found
            worksheet_results.write(row_num, 3, z)  # Z_score

        # Растягиваем ширину колонок
        for col_num, _ in enumerate(df_results.columns):
            max_width = max(
                [len(str(value)) for value in df_results.iloc[:, col_num]] + [len(df_results.columns[col_num])])
            worksheet_results.set_column(col_num, col_num, max_width + 2)  # Добавляем небольшой отступ

        # Вставляем график в Excel со столбца K
        worksheet_results.insert_image('K1', image_path, {'x_scale': 0.5, 'y_scale': 0.5})

    # Удаляем временный файл изображения
    import os
    os.remove(image_path)

    "# 7.3 Оценка коэффициента искажения"

    import pandas as pd
    import numpy as np

    # Расчет значений
    result = df.loc[df['Сумма'] >= 10]
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

    # Запись в файл Excel с настройками
    with pd.ExcelWriter(os.path.join(output_folder, 'Оценка коэффициента искажения.xlsx'),
                        engine='xlsxwriter') as writer:
        df_output.to_excel(writer, sheet_name='Анализ искажения', index=False, startrow=1)

        # Доступ к workbook и worksheet
        workbook = writer.book
        worksheet = writer.sheets['Анализ искажения']

        # Форматирование
        center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # Установка ширины столбцов по максимальной длине значений
        for i, col in enumerate(df_output.columns):
            max_len = max(df_output[col].astype(str).map(len).max(), len(col)) + 2  # Добавляем небольшой отступ
            worksheet.set_column(i, i, max_len, center_format)

        # Добавление заголовка
        worksheet.write(0, 0, 'Анализ искажения данных', workbook.add_format({'bold': True, 'align': 'center'}))
        worksheet.merge_range(0, 0, 0, len(df_output.columns) - 1, 'Анализ искажения данных', center_format)

    """Кластеризация. Этап 1"""
    update_log("Подготовка к кластеризации...")
    df = test_overwrite
    print(df)
    import math
    from datetime import datetime
    # Вероятность появления цифры в первом разряде

    def p_first(d1):
        return math.log10(1 + 1 / d1)

    # Вероятность появления цифры во втором разряде

    def p_second(d2):
        s = 0
        for k in range(1, 10):
            s += math.log10(1 + 1 / (10 * k + d2))
        return s

    # Список, содержащий вероятности появления цифр 1-9 в первом #разряде

    first_teor = [p_first(d1) for d1 in range(1, 10)]
    df_duplicates = df.groupby(["СчетДт", "СчетКт", "Сумма"], as_index=False).count().sort_values(
        by="Организация",
        ascending=False)

    df.loc[:, 'first'] = df['Сумма'].apply(lambda x: int(str(x)[0]))
    print("DKDKDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    print(df.groupby('first')['Сумма'].count())

    # Список, содержащий частоты появления первых цифр в выборке

    first_real = df.groupby('first')['Сумма'].count() / len(df)

    # Расчет среднего абсолютного отклонения

    def MAD(AP, EP, k):
        s = 0
        for i in range(0, k - 1):
            s += abs(AP[i] - EP[i])
        return s / k

    mad = MAD(list(first_real), first_teor, 9)

    # Z-статистика

    def z_stat(AP, EP, N):
        chisl = abs(AP - EP)
        znam = ((EP * (1 - EP)) / N) ** 0.5
        if 1 / (2 * N) < chisl:
            chisl -= 1 / (2 * N)
        return chisl / znam

    # Z-Тест 1 цифры

    z_stats = []
    for i in range(9):
        z_stats.append(z_stat(list(first_real)[i], first_teor[i], len(df)))

    # Расчет хи-квадрат

    def chi2(AC, EC, N):
        k = len(AC)
        chi = 0
        for i in range(k):
            chi += (AC[i] * N - EC[i] * N) ** 2 / EC[i] * N
        return chi

    chi_stat = chi2(list(first_real), first_teor, len(df))

    # Добавление в исходный датафрейм столбца со значениями z-#статистик

    df_first_stats = pd.DataFrame({"first": list(range(1, 10)), "z-stat first": z_stats})
    df = df.merge(df_first_stats, on='first')

    # Z-Тест 2 цифры

    df.loc[:, 'second'] = df['Сумма'].apply(lambda x: int(str(x)[1]))

    second_real = df.groupby('second')['Сумма'].count() / len(df)

    second_teor = [p_second(d1) for d1 in range(0, 10)]

    z_stat_sec = []
    for i in range(10):
        z_stat_sec.append(z_stat(list(second_real)[i], second_teor[i], len(df)))

    df_second_stats = pd.DataFrame({"second": list(range(0, 10)), "z-stat second": z_stat_sec})
    df = df.merge(df_second_stats, on='second')

    # Тест первых двух цифр

    df.loc[:, 'first_two'] = df['Сумма'].apply(lambda x: int(str(x)[:2]))

    two_teor = [p_first(d1) for d1 in range(10, 100)]
    two_real = df.groupby('first_two')['Сумма'].count() / len(df)

    z_stat_two = []
    for i in range(90):
        z_stat_two.append(z_stat(list(two_real)[i], two_teor[i], len(df)))

    df_first_two_stats = pd.DataFrame({"first_two": list(range(10, 100)), "z-stat first_two": z_stat_two})
    df = df.merge(df_first_two_stats, on='first_two')

    # Тест суммирования
    two_real = df.groupby('first_two')['Сумма'].sum() / df['Сумма'].sum()

    df_abs_delta = pd.DataFrame({"first_two": list(range(10, 100)), "sum_frequency": list(two_real)})
    df = df.merge(df_abs_delta, on='first_two')

    # Тест второго порядка

    df_cur = df.sort_values(by='Сумма')
    df_cur.loc[:, 'two'] = df_cur['Сумма'].diff() * 10
    df_cur.dropna(subset=['Сумма'], inplace=True)
    df_cur = df_cur[df_cur['two'] > 10]
    df_cur.loc[:, 'two'] = df_cur['two'].apply(lambda x: int(str(x)[:2]))
    df_cur.shape

    df_z_stat_second_diff = pd.DataFrame({"two": list(range(10, 100)), "z_stat_second_diff": z_stat_two})

    df_cur.head()

    df_cur = df_cur.merge(df_z_stat_second_diff, on="two")

    ind = df_cur.index
    df.loc[ind, "z_stat_second_diff"] = df_cur["z_stat_second_diff"]

    # TODO отсюда продолжить перенос
    # Z-Тест последних двух цифр

    df_cur = df

    df_cur.loc[:, 'last_two'] = df_cur['Сумма'].apply(lambda x: int(str(int(round((x * 100), 0)))[-2:]))

    two_real = df_cur.groupby('last_two')['Сумма'].count() / len(df_cur)

    two_teor = [0.01 for i in range(100)]

    z_stats = []
    for i in range(100):
        z_stats.append(z_stat(list(two_real)[i], two_teor[i], len(df_cur)))

    mad = MAD(list(two_real), two_teor, 100)

    df_last_two = pd.DataFrame({"last_two": list(range(0, 100)), "z_stat_last_two": z_stats})
    df_cur = df_cur.merge(df_last_two, on='last_two')

    def a_socr(a):
        return 10 * a / (10 ** int(math.log(a, 10)))

    df_cur['two'] = df_cur['Сумма'].apply(lambda x: a_socr(x))

    # Добавляется столбец с частотой счета Дт (синтетический счет)

    df_cur["СинтСчетДт"] = df_cur["СчетДт"].apply(lambda x: str(x)[:2])
    df_dt = (df_cur.groupby("СинтСчетДт").count() / len(df_cur))
    df_dt = df_dt.rename(columns={"Организация": "Частота счета Дт"})
    df_dt = df_dt["Частота счета Дт"]
    df_cur = df_cur.merge(df_dt, on='СинтСчетДт')

    # Добавляется столбец с частотой счета Кт (синтетический счет)

    df_cur["СинтСчетКт"] = df_cur["СчетКт"].apply(lambda x: str(x)[:2])
    df_kt = (df_cur.groupby("СинтСчетКт").count() / len(df_cur))
    df_kt = df_kt.rename(columns={"Организация": "Частота счета Кт"})
    df_kt = df_kt["Частота счета Кт"]
    df_cur = df_cur.merge(df_kt, on='СинтСчетКт')

    # Добавляется столбец с частотой проводки (по синтетическим счетам)

    df_cur["Проводка"] = df_cur["СинтСчетДт"] + "-" + df_cur["СинтСчетКт"]
    df_pr = (df_cur.groupby("Проводка").count() / len(df_cur))
    df_pr = df_pr.rename(columns={"Организация": "Частота проводки"})
    df_pr = df_pr["Частота проводки"]
    df_cur = df_cur.merge(df_pr, on="Проводка")

    # Добавляется столбец с частотой автора операции

    df_au = df_cur.groupby("АвторОперации").count() / len(df_cur)
    df_au = df_au.rename(columns={"Организация": "Частота автора операции"})
    df_au = df_au["Частота автора операции"]
    df_cur = df_cur.merge(df_au, on='АвторОперации')

    # Ручная проводка (1) или нет (0)

    df_cur["Ручная проводка"] = df_cur["РучнаяКорректировка"].apply(lambda x: 1 if x == "Да" else 0)

    # выходные дни (1), другие дни (0)

    df_cur["Data"] = df_cur["Период"].apply(lambda x: str(x)[:10])
    df_cur["Data"] = df_cur["Data"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df_cur["Data"] = df_cur["Data"].apply(lambda x: datetime.weekday(x))
    df_cur["Data"] = df_cur["Data"].apply(lambda x: 1 if (x == 5 or x == 6) else 0)
    df_cur = df_cur.rename(columns={"Data": "Выходные или рабочие"})

    # Считает количество дублирующихся проводок

    df_duplicates_ = df_duplicates.iloc[:, :4]
    df_duplicates_.rename({"Организация": "Количество дублей"}, axis=1, inplace=True)
    df_cur = df_cur.merge(df_duplicates_, on=["СчетДт", "СчетКт", "Сумма"])

    temp = df_cur.loc[:,
           ["z-stat first", "z-stat second", "z-stat first_two", "sum_frequency", "z_stat_second_diff",
            "z_stat_last_two",
            "Частота суммы", "Частота счета Дт", "Частота счета Кт", "Частота проводки",
            "Частота автора операции",
            "Ручная проводка", "Выходные или рабочие", "Сторно", "Количество дублей"]]

    temp.loc[temp["z_stat_second_diff"].isna(), "z_stat_second_diff"] = -1
    # Считает количество дублирующихся проводок
    df_duplicates_ = df_duplicates.iloc[:, :4]
    df_duplicates_.rename({"Организация": "Количество дублей"}, axis=1, inplace=True)
    df_cur = df_cur.merge(df_duplicates_, on=["СчетДт", "СчетКт", "Сумма"])

    temp.head()
    from sklearn import preprocessing
    scaled = preprocessing.StandardScaler().fit_transform(temp.iloc[:, :6].values)
    scaled_df = pd.DataFrame(scaled, index=temp.iloc[:, :6].index, columns=temp.iloc[:, :6].columns)
    scaled_df.head()
    plt.close('all')  # Закрывает все открытые графики

    import os
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    import sklearn.metrics as metrics
    update_log("Построение силуэта...")
    # Настройки для графиков
    plt.rcParams["figure.figsize"] = (8, 4)

    # Список возможных значений кластеров
    x = [i for i in range(2, 6)]
    m = []

    sample_size = int(0.2 * scaled_df.shape[0])
    print(f'sample size = {sample_size}')

    # Подсчёт silhouette_score для каждого количества кластеров
    for i in tqdm(x):
        labels = KMeans(n_clusters=i, random_state=1000).fit(scaled_df).labels_
        m.append(metrics.silhouette_score(scaled_df, labels, sample_size=sample_size))

    # Найдём количество кластеров с максимальным значением silhouette_score
    best_n_clusters = x[m.index(max(m))]

    print(f'Оптимальное количество кластеров: {best_n_clusters}')

    # Строим график зависимости silhouette_score от количества кластеров
    plt.plot(x, m, 'r-')
    plt.xticks(ticks=x, labels=[int(i) for i in x])
    plt.xlabel('Количество кластеров')
    plt.ylabel('Значение метрики')
    plt.title('Зависимость значения метрики от количества кластеров')

    # Сохраняем график
    image_path = 'silhouette_plot.png'
    plt.savefig(image_path)
    plt.close('all')  # Закрывает все открытые графики

    # Экспорт данных в Excel
    excel_file = os.path.join(output_folder, 'Силуэт (этап 1).xlsx')
    sheet_name = 'Силуэт'

    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Создаем лист для результатов анализа
        worksheet_silhouette = writer.book.add_worksheet(sheet_name)

        # Записываем строку с оптимальным числом кластеров
        worksheet_silhouette.write(0, 0, f'Оптимальное количество кластеров: {best_n_clusters}')

        # Записываем результаты анализа в Excel
        worksheet_silhouette.write(2, 0, 'Количество кластеров')
        worksheet_silhouette.write(2, 1, 'Silhouette Score')

        for row_num, (n_clusters, score) in enumerate(zip(x, m), start=3):
            worksheet_silhouette.write(row_num, 0, n_clusters)
            worksheet_silhouette.write(row_num, 1, score)

        # Вставляем график в Excel
        worksheet_silhouette.insert_image('C1', image_path, {'x_scale': 1, 'y_scale': 1})

        # Автовыравнивание ширины столбцов A и B
        worksheet_silhouette.set_column(0, 0, max([len(str(n)) for n in x] + [len('Количество кластеров')]))
        worksheet_silhouette.set_column(1, 1, max([len(f'{s:.3f}') for s in m] + [len('Silhouette Score')]))
    # Удаляем временные файлы изображений, если они существуют
    if os.path.exists(image_path):
        os.remove(image_path)

    # TODO

    print(f"График силуэта и результаты анализа успешно сохранены в '{excel_file}'.")

    # Применение KMeans с оптимальным количеством кластеров
    if "Class" in scaled_df.columns:
        scaled_df.drop(columns=["Class"], inplace=True)

    km = KMeans(n_clusters=best_n_clusters, random_state=1000)

    scaled_df["Class"] = km.fit_predict(scaled_df)
    scaled_df["Class"] = scaled_df["Class"] + 1

    grouped_count = scaled_df.groupby("Class").count()["z-stat first"]
    print(grouped_count)

    print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    print(temp)

    temp["Class"] = scaled_df["Class"]

    mean_temp = temp.groupby("Class").mean()
    # Построение графика
    scaled_df.groupby("Class").mean().T.plot(grid=True, figsize=(15, 10),  # Увеличиваем высоту графика
                                             rot=90,
                                             xticks=range(len(scaled_df.columns) - 1),
                                             style='o-', linewidth=4, markersize=12)

    plt.legend(fontsize=30)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=10)  # Уменьшаем шрифт оси y
    plt.tight_layout()  # Подбираем отступы

    # Сохраняем график
    image_path_z_stats = 'z_stats.png'
    plt.savefig(image_path_z_stats)
    plt.close('all')  # Закрывает все открытые графики

    # Добавляем лист с результатами в существующий файл z_stats_report.xlsx
    sheet_name = 'Статистики'
    excel_file = os.path.join(output_folder, 'Z-статистики.xlsx')

    # Удаляем временные файлы изображений, если они существуют
    if os.path.exists(image_path_z_stats):
        os.remove(image_path_z_stats)

    print(f"Лист 'Статистики' добавлен к файлу '{excel_file}'. Файл готов для скачивания.")

    temp.groupby("Class").mean()

    # построение главных компонент

    from sklearn.decomposition import PCA
    update_log("Построение главных компонент...")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_df.iloc[:, :-1])
    principalDf = pd.DataFrame(data=principalComponents)

    principalDf = principalDf.rename(columns={0: "PC1", 1: "PC2"})
    principalDf["Class"] = scaled_df["Class"]

    principalDf.head()

    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    import sklearn.metrics as metrics

    # Визуализация главных компонент
    for i in range(1, best_n_clusters + 1):
        data = principalDf[principalDf["Class"] == i]
        plt.plot(data.PC1, data.PC2, 'o', label=f'Класс {i}')
    plt.legend()
    plt.title("Главные компоненты")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")

    # Сохраняем график
    image_path_pca = 'pca_plot.png'
    plt.savefig(image_path_pca)
    plt.close('all')  # Закрывает все открытые графики

    # Добавляем лист с результатами PCA в существующий файл clustering_report.xlsx
    sheet_name = 'Метод ГК'
    excel_file = os.path.join(output_folder, 'Главные компоненты.xlsx')
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Создаем новый лист "Метод ГК"
        worksheet_pca = writer.book.add_worksheet(sheet_name)
        # Вставляем график главных компонент в Excel
        worksheet_pca.insert_image('A1', image_path_pca, {'x_scale': 1, 'y_scale': 1})

    # Удаляем временные файлы изображений, если они существуют
    if os.path.exists(image_path_pca):
        os.remove(image_path_pca)

    print(f"Лист 'Метод ГК' добавлен к файлу '{excel_file}'. Файл готов для скачивания.")

    # Анализ выбросов

    from sklearn.ensemble import IsolationForest
    update_log("Поиск аномального класса по Isolation Forest...")
    anomaly_labels = IsolationForest().fit_predict(scaled_df.drop(["Class"], axis=1))

    scaled_df["IsoLabels"] = anomaly_labels

    # Список для хранения отношений z-stat first
    ratios = {}

    # Получаем уникальные классы
    classes = scaled_df["Class"].unique()

    # Проходим по каждому классу и вычисляем отношение
    for cls in classes:
        # Получаем count z-stat first для текущего класса
        z_stat_pos = scaled_df[(scaled_df["Class"] == cls) & (scaled_df["IsoLabels"] == 1)][
            "z-stat first"].count()
        z_stat_neg = scaled_df[(scaled_df["Class"] == cls) & (scaled_df["IsoLabels"] == -1)][
            "z-stat first"].count()

        # Проверяем, чтобы z_stat_neg не было нулем
        if z_stat_neg != 0:
            ratio = z_stat_pos / z_stat_neg
            ratios[cls] = ratio
        else:
            ratio = z_stat_pos
            ratios[cls] = ratio

    # Находим класс с минимальным отношением
    if ratios:
        anomaly_class_iso = min(ratios, key=ratios.get)  # Класс с минимальным отношением
    else:
        anomaly_class_iso = None  # Если нет классов

    print(f'Аномальный класс по Isolation Forest: {anomaly_class_iso}')

    from sklearn.covariance import EllipticEnvelope
    update_log("Поиск аномального класса по Elliptic Envelope...")
    anomaly_labels_el = EllipticEnvelope().fit_predict(scaled_df.drop(["Class", "IsoLabels"], axis=1))

    scaled_df["ElLabels"] = anomaly_labels_el
    scaled_df.groupby(["ElLabels", "IsoLabels", "Class"]).count()

    ratios_el = {}

    # Проходим по каждому классу и вычисляем отношение для Elliptic Envelope
    for cls in classes:
        # Получаем count z-stat first для текущего класса
        z_stat_pos = scaled_df[(scaled_df["Class"] == cls) & (scaled_df["ElLabels"] == 1)][
            "z-stat first"].count()
        z_stat_neg = scaled_df[(scaled_df["Class"] == cls) & (scaled_df["ElLabels"] == -1)][
            "z-stat first"].count()

        # Проверяем, чтобы z_stat_neg не было нулем
        if z_stat_neg != 0:
            ratio = z_stat_pos / z_stat_neg
            ratios_el[cls] = ratio
        else:
            ratio = z_stat_pos
            ratios_el[cls] = ratio

    # Находим аномальный класс для Elliptic Envelope
    if ratios_el:
        anomaly_class_el = min(ratios_el, key=ratios_el.get)  # Класс с минимальным отношением
    else:
        anomaly_class_el = None  # Если нет классов

    print(f'Аномальный класс по Elliptic Envelope: {anomaly_class_el}')

    import pandas as pd

    # Группируем данные и заполняем нулями
    grouped_iso = scaled_df.groupby(["IsoLabels", "Class"]).count().reindex(
        pd.MultiIndex.from_product([[-1, 1], classes], names=["IsoLabels", "Class"]), fill_value=0
    )

    grouped_el = scaled_df.groupby(["ElLabels", "Class"]).count().reindex(
        pd.MultiIndex.from_product([[-1, 1], classes], names=["ElLabels", "Class"]), fill_value=0
    )

    # Создаем новый Excel-файл
    excel_file = os.path.join(output_folder, 'Аномальный класс (этап 1).xlsx')
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Записываем результаты для Isolation Forest
        worksheet_iso = writer.book.add_worksheet('Isolation Forest')
        worksheet_iso.write(0, 0, 'IsoLabels')
        worksheet_iso.write(0, 1, 'Класс')
        worksheet_iso.write(0, 2, 'Число объектов')

        # Записываем данные для IsoLabels
        for row_num, (index, row) in enumerate(grouped_iso.iterrows(), start=1):
            worksheet_iso.write(row_num, 0, index[0])  # IsoLabels
            worksheet_iso.write(row_num, 1, index[1])  # Class
            worksheet_iso.write(row_num, 2, row['z-stat first'])  # Count

        # Добавляем информацию об аномальном классе по Isolation Forest
        worksheet_iso.write(len(grouped_iso) + 2, 0, 'Аномальный класс по Isolation Forest:')
        worksheet_iso.write(len(grouped_iso) + 2, 1, anomaly_class_iso)

        # Добавляем разделитель между результатами
        worksheet_iso.write(len(grouped_iso) + 4, 0, '-----')  # Разделитель
        worksheet_iso.write(len(grouped_iso) + 5, 0, 'Elliptic Envelope')  # Заголовок для Elliptic Envelope

        # Записываем результаты для Elliptic Envelope
        worksheet_iso.write(len(grouped_iso) + 6, 0, 'ElLabels')
        worksheet_iso.write(len(grouped_iso) + 6, 1, 'Класс')
        worksheet_iso.write(len(grouped_iso) + 6, 2, 'Число объектов')

        for row_num, (index, row) in enumerate(grouped_el.iterrows(), start=len(grouped_iso) + 7):
            worksheet_iso.write(row_num, 0, index[0])  # ElLabels
            worksheet_iso.write(row_num, 1, index[1])  # Class
            worksheet_iso.write(row_num, 2, row['z-stat first'])  # Count

        # Добавляем информацию об аномальном классе по Elliptic Envelope
        worksheet_iso.write(len(grouped_iso) + len(grouped_el) + 7, 0, 'Аномальный класс по Elliptic Envelope:')
        worksheet_iso.write(len(grouped_iso) + len(grouped_el) + 7, 1, anomaly_class_el)

    print(f'Файл {excel_file} создан с аномальными классами для обоих методов.')

    scaled_df_temp = scaled_df.copy()
    scaled_df = scaled_df.drop(["ElLabels", "IsoLabels"], axis=1)

    import matplotlib.pyplot as plt
    import pandas as pd
    import xlsxwriter
    import os

    # Путь к файлу Excel
    excel_file = os.path.join(output_folder, 'Аномальный класс (графики).xlsx')

    # Создание графиков для каждого класса и сохранение в файлы PNG
    classes = scaled_df_temp["Class"].unique()
    image_files = []

    # Проходим по каждому классу и строим графики
    for cls in classes:
        plt.figure(figsize=(15, 8))

        # Основной график для среднего значения по классу
        scaled_df_temp[scaled_df_temp["Class"] == cls].iloc[:, :-3].mean().T.plot(
            style='o-', linewidth=4, markersize=12, label=f'Среднее по классу {cls}'
        )

        # График для IsoLabels=-1 и ElLabels=-1 для данного класса
        scaled_df_temp[
            (scaled_df_temp["Class"] == cls) &
            (scaled_df_temp["IsoLabels"] == -1) &
            (scaled_df_temp["ElLabels"] == -1)
            ].iloc[:, :-3].mean().T.plot(
            style='o--', linewidth=4, markersize=12, label=f'Аномалии по обоим методам (-1, -1)'
        )

        # График для IsoLabels=1 и ElLabels=1 для данного класса
        scaled_df_temp[
            (scaled_df_temp["Class"] == cls) &
            (scaled_df_temp["IsoLabels"] == 1) &
            (scaled_df_temp["ElLabels"] == 1)
            ].iloc[:, :-3].mean().T.plot(
            style='o--', linewidth=4, markersize=12, label=f'Нет аномалий по обоим методам (1, 1)'
        )

        # Настройки графика
        plt.legend(fontsize=14)
        plt.title(f'График для класса {cls}')
        plt.xlabel('Показатели')
        plt.ylabel('Значение')

        # Добавление сетки
        plt.grid(True, linestyle='--', linewidth=0.5)  # Настройки сетки

        # Сохранение графика в PNG
        image_file = f'class_{cls}_plot.png'
        plt.savefig(image_file)
        image_files.append(image_file)
        plt.close('all')  # Закрывает все открытые графики

    # Создание файла Excel с помощью xlsxwriter
    with xlsxwriter.Workbook(excel_file) as workbook:
        worksheet = workbook.add_worksheet('Графики')

        # Координаты для вставки графиков
        start_row = 0

        # Вставка всех графиков на один лист с уменьшением масштаба
        for cls, image_file in zip(classes, image_files):
            # Вставляем изображение графика на лист, уменьшая масштаб по осям
            worksheet.insert_image(start_row, 0, image_file, {'x_scale': 0.5, 'y_scale': 0.5})

            # Обновляем стартовую строку для следующего графика
            start_row += 20  # Подбираем это значение в зависимости от высоты графиков

    # Удаляем временные файлы с графиками после вставки в Excel
    for image_file in image_files:
        os.remove(image_file)

    print(f'Графики успешно сохранены в файл {excel_file}')

    scaled_df = scaled_df_temp[(scaled_df_temp["ElLabels"] == -1) & (scaled_df_temp["IsoLabels"] == -1) & (
            scaled_df_temp["Class"] == anomaly_class_el)].iloc[:, :-2]

    scaled_df.head()

    # Получаем индексы аномальных объектов из scaled_df
    anomaly_indices_el = scaled_df[scaled_df["Class"] == anomaly_class_el].index

    # Фильтрация исходного DataFrame df по этим индексам
    anomaly_original_rows = df.loc[anomaly_indices_el]

    # Вывод
    anomaly_original_rows.head(10)

    # вывод всех аномальных строк изначального документа
    anomaly_original_rows.to_excel(os.path.join(output_folder, 'Список подозрительных операций (этап 1).xlsx'),
                                   index=False)

    cluster_1_index = list(scaled_df[scaled_df["Class"] == anomaly_class_el].index)

    # df_cur[(df_cur.index.ster_1_isin(cluindex)) & (df_cur["ВидСубконтоДт1"] == "Контрагенты")].groupby("СубконтоДт1").count().sort_values(by="Сумма", ascending = False).head(10)

    df_class_1 = temp[(temp.index.isin(cluster_1_index))]

    temp_class_1 = df_class_1.loc[:,
                   ["z-stat first", "z-stat second", "z-stat first_two", "sum_frequency", "z_stat_second_diff",
                    "Частота суммы", "Частота счета Дт", "Частота счета Кт", "Частота проводки",
                    "Частота автора операции",
                    "Ручная проводка", "Выходные или рабочие", "Сторно", "Количество дублей"]]

    """**3.2 Кластеризация, Этап 2**"""

    # стандартизация данных
    # проведем кластеризацию для объектов аномального класса по оставшимся признакам
    update_log("Построение силуэта аномального класса...")
    from sklearn.preprocessing import StandardScaler

    scaled_ = StandardScaler().fit_transform(temp_class_1.iloc[:, 6:].values)
    scaled_class_1 = pd.DataFrame(scaled_, index=temp_class_1.iloc[:, 6:].index,
                                  columns=temp_class_1.iloc[:, 6:].columns)
    scaled_class_1.head()

    # Силуэт
    plt.close('all')  # Закрывает все открытые графики

    plt.rcParams["figure.figsize"] = (8, 4)

    x = [i for i in range(2, 6)]
    m = []

    sample_size = int(scaled_class_1.shape[0])
    print(f'sample size = {sample_size}')
    for i in tqdm(x):
        print(i)
        labels = KMeans(n_clusters=i, random_state=1000).fit(scaled_class_1).labels_
        print(labels)
        m.append(metrics.silhouette_score(scaled_class_1, labels, sample_size=sample_size))
        print(f'n = {i}, silhouette = {m[-1]}')
    # Найдём количество кластеров с максимальным значением silhouette_score
    best_n_clusters = x[m.index(max(m))]

    # Строим график зависимости silhouette_score от количества кластеров
    plt.plot(x, m, 'r-')
    plt.xticks(ticks=x, labels=[int(i) for i in x])
    plt.xlabel('Количество кластеров')
    plt.ylabel('Значение метрики')
    plt.title('Зависимость значения метрики от количества кластеров')

    # Сохраняем график
    image_path = 'clust2_silhouette_plot.png'
    plt.savefig(image_path)
    plt.close('all')  # Закрывает все открытые графики

    # Экспорт данных в Excel
    excel_file = os.path.join(output_folder, 'Силуэт (этап 2).xlsx')
    sheet_name = 'Силуэт'

    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Создаем лист для результатов анализа
        worksheet_silhouette = writer.book.add_worksheet(sheet_name)

        # Записываем строку с оптимальным числом кластеров
        worksheet_silhouette.write(0, 0, f'Оптимальное количество кластеров: {best_n_clusters}')

        # Записываем результаты анализа в Excel
        worksheet_silhouette.write(2, 0, 'Количество кластеров')
        worksheet_silhouette.write(2, 1, 'Silhouette Score')

        for row_num, (n_clusters, score) in enumerate(zip(x, m), start=3):
            worksheet_silhouette.write(row_num, 0, n_clusters)
            worksheet_silhouette.write(row_num, 1, score)

        # Вставляем график в Excel
        worksheet_silhouette.insert_image('C1', image_path, {'x_scale': 1, 'y_scale': 1})

        # Автовыравнивание ширины столбцов A и B
        worksheet_silhouette.set_column(0, 0, max([len(str(n)) for n in x] + [len('Количество кластеров')]))
        worksheet_silhouette.set_column(1, 1, max([len(f'{s:.3f}') for s in m] + [len('Silhouette Score')]))

    # Удаляем временные файлы изображений, если они существуют
    if os.path.exists(image_path):
        os.remove(image_path)

    print(f"График силуэта и результаты анализа успешно сохранены в '{excel_file}'.")
    plt.close('all')  # Закрывает все открытые графики

    n_clusters = best_n_clusters
    if "Class" in scaled_class_1.columns:
        scaled_class_1.drop(columns=["Class", ], inplace=True)
    km = KMeans(n_clusters=n_clusters, random_state=1000)
    scaled_class_1["Class"] = km.fit_predict(scaled_class_1)
    scaled_class_1["Class"] = scaled_class_1["Class"] + 1

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Удаляем столбец "Class", если он уже есть
    if "Class" in scaled_class_1.columns:
        scaled_class_1.drop(columns=["Class"], inplace=True)

    # Применение KMeans
    km = KMeans(n_clusters=n_clusters, random_state=1000)
    scaled_class_1["Class"] = km.fit_predict(scaled_class_1)
    scaled_class_1["Class"] = scaled_class_1["Class"] + 1  # Нумерация классов с 1

    # Вычисляем количество объектов в каждом классе
    class_counts = scaled_class_1.groupby("Class").count()["Сторно"]
    class_counts = class_counts.reset_index()  # Преобразуем индекс в столбец для удобного вывода

    # Переименовываем столбцы
    class_counts.columns = ['Класс', 'Количество объектов']

    # Рисуем график средних значений по каждому классу
    scaled_class_1.groupby("Class").mean().T.plot(grid=True, figsize=(15, 10), rot=90,
                                                  xticks=range(len(scaled_class_1.columns) - 1), style='o-',
                                                  linewidth=4,
                                                  markersize=12)

    plt.legend(fontsize=30)
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    # Сохраняем график во временный файл
    plt.savefig('cluster_plot.png', bbox_inches='tight')
    plt.close('all')  # Закрывает все открытые графики
    if os.path.exists(image_path):
        os.remove(image_path)
    # Создаем или открываем файл Excel для сохранения
    excel_file = os.path.join(output_folder, 'Кластеризация (этап 2).xlsx')

    # Используем ExcelWriter для работы с xlsxwriter
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Записываем данные по количеству объектов в каждом классе в Excel
        class_counts.to_excel(writer, sheet_name='KMeans Clustering', startrow=0,
                              index=False)  # index=False убирает индекс

        # Доступ к объекту workbook и worksheet для вставки графика
        workbook = writer.book
        worksheet = writer.sheets['KMeans Clustering']

        # Вставляем график ниже данных
        worksheet.insert_image('A10', 'cluster_plot.png', {'x_scale': 0.7, 'y_scale': 0.7})

    print(f'Файл {excel_file} успешно создан с графиком и данными кластеров.')
    # Удаляем временные файлы изображений, если они существуют
    if os.path.exists(image_path):
        os.remove(image_path)

    # Анализ выбросов
    update_log("Поиск аномального класса по Isolation Forest...")
    from sklearn.ensemble import IsolationForest

    anomaly_labels = IsolationForest().fit_predict(scaled_class_1.drop(["Class"], axis=1))

    import pandas as pd

    scaled_class_1["IsoLabels"] = anomaly_labels

    # Список для хранения отношений Сторно
    ratios = {}

    # Определяем порядок классов
    class_order = sorted(scaled_class_1["Class"].unique())  # [1, 2, 3, 4, 5]

    # Проходим по каждому классу и вычисляем отношение
    for cls in class_order:
        z_stat_pos = scaled_class_1[(scaled_class_1["Class"] == cls) & (scaled_class_1["IsoLabels"] == 1)][
            "Сторно"].count()
        z_stat_neg = scaled_class_1[(scaled_class_1["Class"] == cls) & (scaled_class_1["IsoLabels"] == -1)][
            "Сторно"].count()

        if z_stat_neg != 0:
            ratio = z_stat_pos / z_stat_neg
            ratios[cls] = ratio
        else:
            ratio = z_stat_pos
            ratios[cls] = ratio

    # Находим класс с минимальным отношением
    if ratios:
        anomaly_class_iso = min(ratios, key=ratios.get)
    else:
        anomaly_class_iso = None

    print(f'Аномальный класс по Isolation Forest: {anomaly_class_iso}')

    from sklearn.covariance import EllipticEnvelope
    update_log("Поиск аномального класса по Elliptic Envelope...")
    anomaly_labels_el = EllipticEnvelope().fit_predict(scaled_class_1.drop(["Class", "IsoLabels"], axis=1))

    scaled_class_1["ElLabels"] = anomaly_labels_el

    ratios_el = {}

    for cls in class_order:
        z_stat_pos = scaled_class_1[(scaled_class_1["Class"] == cls) & (scaled_class_1["ElLabels"] == 1)][
            "Сторно"].count()
        z_stat_neg = scaled_class_1[(scaled_class_1["Class"] == cls) & (scaled_class_1["ElLabels"] == -1)][
            "Сторно"].count()

        if z_stat_neg != 0:
            ratio = z_stat_pos / z_stat_neg
            ratios_el[cls] = ratio
        else:
            ratio = z_stat_pos
            ratios_el[cls] = ratio

    if ratios_el:
        anomaly_class_el = min(ratios_el, key=ratios_el.get)
    else:
        anomaly_class_el = None

    print(f'Аномальный класс по Elliptic Envelope: {anomaly_class_el}')

    # Группируем данные и заполняем нулями, устанавливая порядок классов
    grouped_iso = scaled_class_1.groupby(["IsoLabels", "Class"]).count().reindex(
        pd.MultiIndex.from_product([[-1, 1], class_order], names=["IsoLabels", "Class"]), fill_value=0
    )

    grouped_el = scaled_class_1.groupby(["ElLabels", "Class"]).count().reindex(
        pd.MultiIndex.from_product([[-1, 1], class_order], names=["ElLabels", "Class"]), fill_value=0
    )

    # Создаем новый Excel-файл
    excel_file = os.path.join(output_folder, 'Аномальный класс (этап 2).xlsx')
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Записываем результаты для Isolation Forest
        worksheet_iso = writer.book.add_worksheet('Isolation Forest')
        worksheet_iso.write(0, 0, 'IsoLabels')
        worksheet_iso.write(0, 1, 'Класс')
        worksheet_iso.write(0, 2, 'Число объектов')

        for row_num, (index, row) in enumerate(grouped_iso.iterrows(), start=1):
            worksheet_iso.write(row_num, 0, index[0])  # IsoLabels
            worksheet_iso.write(row_num, 1, index[1])  # Class
            worksheet_iso.write(row_num, 2, row['Сторно'])  # Count

        # Добавляем информацию об аномальном классе по Isolation Forest
        worksheet_iso.write(len(grouped_iso) + 2, 0, 'Аномальный класс по Isolation Forest:')
        worksheet_iso.write(len(grouped_iso) + 2, 1, anomaly_class_iso)

        # Добавляем разделитель между результатами
        worksheet_iso.write(len(grouped_iso) + 4, 0, '-----')  # Разделитель
        worksheet_iso.write(len(grouped_iso) + 5, 0, 'Elliptic Envelope')  # Заголовок для Elliptic Envelope

        # Записываем результаты для Elliptic Envelope
        worksheet_iso.write(len(grouped_iso) + 6, 0, 'ElLabels')
        worksheet_iso.write(len(grouped_iso) + 6, 1, 'Класс')
        worksheet_iso.write(len(grouped_iso) + 6, 2, 'Число объектов')

        for row_num, (index, row) in enumerate(grouped_el.iterrows(), start=len(grouped_iso) + 7):
            worksheet_iso.write(row_num, 0, index[0])  # ElLabels
            worksheet_iso.write(row_num, 1, index[1])  # Class
            worksheet_iso.write(row_num, 2, row['Сторно'])  # Count

        # Добавляем информацию об аномальном классе по Elliptic Envelope
        worksheet_iso.write(len(grouped_iso) + len(grouped_el) + 7, 0, 'Аномальный класс по Elliptic Envelope:')
        worksheet_iso.write(len(grouped_iso) + len(grouped_el) + 7, 1, anomaly_class_el)

    print(f'Файл {excel_file} создан с аномальными классами для обоих методов.')
    plt.close('all')
    update_log("Вывод подозрительных операций...")
    # Получаем индексы аномальных объектов из scaled_class_1
    anomaly_indices_el = scaled_class_1[scaled_class_1["Class"] == anomaly_class_el].index

    # Фильтрация исходного DataFrame df по этим индексам
    anomaly_original_rows = df.loc[anomaly_indices_el]

    # вывод всех аномальных строк изначального документа
    anomaly_original_rows.to_excel(os.path.join(output_folder, 'Подозрительные операции (этап 2).xlsx'), index=False)

    messagebox.showinfo("Результат", f"Тест завершен. Проверьте вывод в папку {output_folder} для деталей.")

    # После завершения анализа скрываем все кнопки, кроме "Выход" и "Провести новый анализ"
    start_button.pack_forget()
    load_osv_button.pack_forget()
    start_analysis_button.pack_forget()
    update_log("Анализ завершён.")
    log_label.pack_forget()  # Можно скрыть метку, если она больше не нужна

    # Показать кнопку "Провести новый анализ"
    restart_button.pack(side=tk.LEFT, padx=10, pady=10)


# Функция для выхода из программы
def exit_program():
    window.quit()


# Функция для перезапуска окна приложения
def restart_program():
    window.destroy()
    create_main_window()


import os
import tkinter as tk
from tkinter import font as tkfont


def create_main_window():
    global window, log_label, load_osv_button, start_analysis_button, start_button, restart_button, risk_entry, company_entry
    # Создаем основное окно
    window = tk.Tk()
    window.title("FraudFounder")
    window.geometry("500x400")
    window.configure(bg="#f0f4fc")  # Светло-голубой фон

    # Получаем путь к текущей директории
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Путь к файлу иконки
    icon_path = os.path.join(base_dir, 'resources', 'fraud.ico')

    # Устанавливаем иконку
    window.iconbitmap(icon_path)

    # Создаем кастомный шрифт (если шрифт "Jury" установлен)
    try:
        custom_font = tkfont.Font(family="Jury", size=12, weight="bold")
    except:
        custom_font = tkfont.Font(family="Helvetica", size=12, weight="bold")  # Альтернативный шрифт

    # Функция для валидации числового ввода
    def validate_numeric_input(value_if_allowed):
        if value_if_allowed == "" or value_if_allowed.replace('.', '', 1).isdigit():
            return True
        else:
            return False

    # Регистрируем валидацию
    vcmd = (window.register(validate_numeric_input), '%P')

    # Поле ввода для "Ожидаемый риск мошенничества"
    risk_label = tk.Label(window, text="Ожидаемый риск мошенничества", anchor="w", bg="#f0f4fc", fg="#276fbf",
                          font=custom_font)
    risk_label.pack(pady=5, padx=10, fill="x")
    risk_entry = tk.Entry(window, validate="key", validatecommand=vcmd, font=custom_font, relief="flat", bg="#d4e5fa")
    risk_entry.insert(0, "0.1")  # Устанавливаем дефолтное значение
    risk_entry.pack(pady=5, padx=10, fill="x")

    # Поле ввода для "Название компании"
    company_label = tk.Label(window, text="Название компании", anchor="w", bg="#f0f4fc", fg="#276fbf", font=custom_font)
    company_label.pack(pady=5, padx=10, fill="x")
    company_entry = tk.Entry(window, font=custom_font, relief="flat", bg="#d4e5fa")
    company_entry.pack(pady=5, padx=10, fill="x")

    # Стили кнопок
    button_style = {
        "bg": "#509ee3",
        "fg": "white",
        "font": custom_font,
        "activebackground": "#276fbf",
        "relief": "flat",  # Убираем объем с кнопок
        "bd": 0  # Убираем границы
    }

    # Кнопка для загрузки первого файла
    start_button = tk.Button(window, text="Загрузить первый файл (ЖО)", command=load_first_file, **button_style)
    start_button.pack(pady=10, padx=10, fill='x')

    # Кнопка для загрузки второго файла
    load_osv_button = tk.Button(window, text="Загрузить второй файл (ОСВ)", command=load_second_file, **button_style)
    load_osv_button.pack(pady=10, padx=10, fill='x')
    load_osv_button.config(state=tk.DISABLED)

    # Кнопка для начала анализа
    start_analysis_button = tk.Button(window, text="Начать анализ", command=start_analysis, **button_style)
    start_analysis_button.pack(pady=10, padx=10, fill='x')
    start_analysis_button.pack_forget()

    # Лог для отображения текущих действий
    log_label = tk.Label(window, text="Лог действий будет здесь", anchor="w", bg="#f0f4fc", fg="#276fbf",
                         font=custom_font)
    log_label.pack(side="bottom", fill="x", padx=10, pady=10)

    # Кнопка для проведения нового анализа (изначально скрыта)
    restart_button = tk.Button(window, text="Провести новый анализ", command=restart_program, **button_style)
    restart_button.pack_forget()

    # Кнопка выхода (с красным цветом для выхода)
    exit_button = tk.Button(window, text="Выход", command=exit_program, bg="#ff5f56", fg="white", font=custom_font,
                            activebackground="#e63946", relief="flat", bd=0)
    exit_button.pack(side="right", padx=10, pady=10)

    window.mainloop()


# Запуск приложения
create_main_window()
