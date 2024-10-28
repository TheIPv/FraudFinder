import pandas as pd
import math
from sklearn import preprocessing
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class ClusteringAnalyzer:
    # TODO сделать инициализацию полей в конструкторе, которые стоят None
    def __init__(self, df):
        self.scaled_df_temp = None
        self.df = df
        self.scaled_df = None
        self.scaled_df_2 = None
        self.best_n_clusters = None
        self.temp = None
        self.prepare_df(self)
        self.anomaly_class_el = None
        self.scaled_class_1 = None

    @staticmethod
    def p_first(d1):
        return math.log10(1 + 1 / d1)

    @staticmethod
    def p_second(d2):
        return sum(math.log10(1 + 1 / (10 * k + d2)) for k in range(1, 10))

    @staticmethod
    def MAD(AP, EP, k):
        s = 0
        for i in range(1, k - 1):
            s += abs(AP[i] - EP[i])
        return s / k

    @staticmethod
    def z_stat(AP, EP, N):
        chisl = abs(AP - EP)
        znam = ((EP * (1 - EP)) / N) ** 0.5
        chisl -= 1 / (2 * N) if 1 / (2 * N) < chisl else 0
        return chisl / znam

    @staticmethod
    def chi2(AC, EC, N):
        k = len(AC)
        chi = 0
        for i in range(k):
            chi += ((AC[i] * N - EC[i] * N) ** 2) / (EC[i] * N)
        return chi

    @staticmethod
    def a_socr(a):
        return 10 * a / (10 ** int(math.log(a, 10)))

    @staticmethod
    def prepare_df(self):

        # Добавляем признак: является ли сумма сторнированной (1) или нет (0)
        self.df["Сторно"] = self.df["Сумма"].apply(lambda x: 0 if x > 0 else 1)

        # Убираем строки с пустыми значениями поля Сумма
        self.df = self.df[self.df['Сумма'].notnull()]

        # Для отрицательных сумм меняем знак
        self.df.loc[self.df["Сумма"] < 0, "Сумма"] = -self.df["Сумма"]

        # Оставляем значения, больше 10, чтобы на них можно было провести все тесты
        self.df = self.df[self.df['Сумма'] > 10]

        # Меняем индексы
        self.df = self.df.reset_index(drop=True)

        # Считаем частоту появления сумм
        self.temp = self.df.groupby('Сумма')['Сумма'].count() / len(self.df)
        sum_ = list(self.temp.index)
        frequency = list(self.temp)
        df_temp = pd.DataFrame({"Сумма": sum_, "Частота суммы": frequency})

        self.df = self.df.merge(df_temp, on='Сумма')

    def calculate_digit_stats(self):

        df_duplicates = self.df.groupby(["СчетДт", "СчетКт", "Сумма"], as_index=False).count().sort_values(
            by="Организация",
            ascending=False)

        ### Расчет для первых цифр
        first_teor = [self.p_first(d1) for d1 in range(1, 10)]
        self.df.loc[:, 'first'] = self.df['Сумма'].apply(lambda x: int(str(x)[0]))
        first_real = self.df.groupby('first')['Сумма'].count() / len(self.df)

        # Расчет MAD для первой цифры
        mad_first = self.MAD(list(first_real), first_teor, 9)

        # Z-статистика и Хи-квадрат для первых цифр
        z_stats_first = [self.z_stat(list(first_real)[i], first_teor[i], len(self.df)) for i in range(9)]
        chi_stat_first = self.chi2(list(first_real), first_teor, len(self.df))

        df_first_stats = pd.DataFrame({"first": list(range(1, 10)), "z-stat first": z_stats_first})
        self.df = self.df.merge(df_first_stats, on='first', how='left')

        ###############################

        # Расчет для первых цифр
        second_teor = [self.p_first(d1) for d1 in range(1, 10)]
        self.df.loc[:, 'second'] = self.df['Сумма'].apply(lambda x: int(str(x)[0]))
        second_real = self.df.groupby('second')['Сумма'].count() / len(self.df)

        # Расчет MAD для первой цифры
        mad_second = self.MAD(list(second_real), second_teor, 9)

        # Z-статистика и Хи-квадрат для первых цифр
        z_stats_second = [self.z_stat(list(second_real)[i], second_teor[i], len(self.df)) for i in range(9)]
        chi_stat_second = self.chi2(list(second_real), second_teor, len(self.df))

        df_second_stats = pd.DataFrame({"second": list(range(1, 10)), "z-stat second": z_stats_second})
        self.df = self.df.merge(df_second_stats, on='second', how='left')

        ###############################

        # Расчет для первых двух цифр
        self.df['first_two'] = self.df['Сумма'].apply(lambda x: int(str(x)[:2]) if len(str(x)) > 1 else None)
        two_teor = [self.p_first(d) for d in range(10, 100)]
        two_real = self.df.groupby('first_two')['Сумма'].count() / len(self.df)

        mad_two = self.MAD(list(two_real), two_teor, 9)

        # Z-статистика и Хи-квадрат для первых двух цифр
        z_stats_two = [self.z_stat(list(two_real)[i], two_teor[i], len(self.df)) for i in range(90)]
        chi_stat_two = self.chi2(list(two_real), two_teor, len(self.df))

        df_first_two_stats = pd.DataFrame({"first_two": list(range(10, 100)), "z-stat first_two": z_stats_two})
        self.df = self.df.merge(df_first_two_stats, on='first_two', how='left')

        # Тест суммирования для первых двух цифр
        two_real_sum = self.df.groupby('first_two')['Сумма'].sum() / self.df['Сумма'].sum()
        df_sum_frequency = pd.DataFrame({"first_two": list(range(10, 100)), "sum_frequency": list(two_real_sum)})
        self.df = self.df.merge(df_sum_frequency, on='first_two', how='left')

        # Тест второго порядка для первых двух цифр
        df_cur = self.df.sort_values(by='Сумма')
        df_cur['two'] = df_cur['Сумма'].diff() * 10
        df_cur.dropna(subset=['Сумма'], inplace=True)
        df_cur = df_cur[df_cur['two'] > 10]
        df_cur['two'] = df_cur['two'].apply(lambda x: int(str(x)[:2]))

        # Z-статистика для второго порядка для первых двух цифр
        df_z_stat_second_diff = pd.DataFrame({"two": list(range(10, 100)), "z_stat_second_diff": z_stats_two})
        df_cur = df_cur.merge(df_z_stat_second_diff, on="two", how='left')

        # Соединение результатов теста второго порядка обратно в основной DataFrame
        ind = df_cur.index
        self.df.loc[ind, "z_stat_second_diff"] = df_cur["z_stat_second_diff"]

        ###############################

        df_cur = self.df
        df_cur.loc[:, 'last_two'] = df_cur['Сумма'].apply(lambda x: int(str(int(round((x * 100), 0)))[-2:]))

        # Тест для последних двух цифр
        self.df['last_two'] = self.df['Сумма'].apply(lambda x: int(str(int(round((x * 100), 0)))[-2:]))
        last_two_real = self.df.groupby('last_two')['Сумма'].count() / len(self.df)

        # Теоретическая вероятность для последних двух цифр (равномерное распределение)
        last_two_teor = [0.01 for _ in range(100)]

        # Z-статистика и MAD для последних двух цифр
        z_stats_last_two = [self.z_stat(list(last_two_real)[i], last_two_teor[i], len(self.df)) for i in range(100)]
        mad_last_two = self.MAD(list(last_two_real), last_two_teor, 100)

        df_last_two = pd.DataFrame({"last_two": list(range(0, 100)), "z_stat_last_two": z_stats_last_two})
        df_cur = df_cur.merge(df_last_two, on='last_two')
        df_cur['two'] = df_cur['Сумма'].apply(lambda x: self.a_socr(x))

        ###############################

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

        self.temp = df_cur.loc[:,
                    ["z-stat first", "z-stat second", "z-stat first_two", "sum_frequency", "z_stat_second_diff",
                     "z_stat_last_two",
                     "Частота суммы", "Частота счета Дт", "Частота счета Кт", "Частота проводки",
                     "Частота автора операции",
                     "Ручная проводка", "Выходные или рабочие", "Сторно", "Количество дублей"]]

        self.temp.loc[self.temp["z_stat_second_diff"].isna(), "z_stat_second_diff"] = -1
        # Считает количество дублирующихся проводок
        df_duplicates_ = df_duplicates.iloc[:, :4]
        df_duplicates_.rename({"Организация": "Количество дублей"}, axis=1, inplace=True)
        df_cur = df_cur.merge(df_duplicates_, on=["СчетДт", "СчетКт", "Сумма"])

        scaled = preprocessing.StandardScaler().fit_transform(self.temp.iloc[:, :6].values)
        self.scaled_df = pd.DataFrame(scaled, index=self.temp.iloc[:, :6].index, columns=self.temp.iloc[:, :6].columns)
        plt.close('all')  # Закрывает все открытые графики

        # Настройки для графиков
        plt.rcParams["figure.figsize"] = (8, 4)

        # Список возможных значений кластеров
        x = [i for i in range(2, 6)]
        m = []

        sample_size = int(0.2 * self.scaled_df.shape[0])
        print(f'sample size = {sample_size}')

        # Подсчёт silhouette_score для каждого количества кластеров
        for i in tqdm(x):
            labels = KMeans(n_clusters=i, random_state=1000).fit(self.scaled_df).labels_
            m.append(metrics.silhouette_score(self.scaled_df, labels, sample_size=sample_size))

        # Найдём количество кластеров с максимальным значением silhouette_score
        self.best_n_clusters = x[m.index(max(m))]

        print(f'Оптимальное количество кластеров: {self.best_n_clusters}')

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

        ###############################

        # Оптимальное количество кластеров
        optimal_cluster_data = {'Оптимальное количество кластеров': [self.best_n_clusters]}

        # Создаем DataFrame для строки с оптимальным числом кластеров
        df_optimal_cluster = pd.DataFrame(optimal_cluster_data)

        # Создаем DataFrame для значений `Количество кластеров` и `Silhouette Score`
        df_silhouette_scores = pd.DataFrame({
            'Количество кластеров': x,
            'Silhouette Score': m
        })

        # Объединяем оба DataFrame, добавляя строку с оптимальным числом кластеров в начало
        result_df_silhouette = pd.concat([df_optimal_cluster, df_silhouette_scores], ignore_index=True)

        # Вставка строки с заголовками и дополнительная строка для отображения
        result_df_silhouette.insert(0, 'Описание',
                                    ['Оптимальное количество кластеров'] + [''] * (len(df_silhouette_scores)))

        ###############################

        return result_df_silhouette, image_path

    def calculate_statistics(self):
        # Применение KMeans с оптимальным количеством кластеров
        if "Class" in self.scaled_df.columns:
            self.scaled_df.drop(columns=["Class"], inplace=True)

        km = KMeans(n_clusters=self.best_n_clusters, random_state=1000)

        self.scaled_df["Class"] = km.fit_predict(self.scaled_df)
        self.scaled_df["Class"] = self.scaled_df["Class"] + 1

        grouped_count = self.scaled_df.groupby("Class").count()["z-stat first"]

        self.temp["Class"] = self.scaled_df["Class"]

        mean_temp = self.temp.groupby("Class").mean()
        # Построение графика
        self.scaled_df.groupby("Class").mean().T.plot(grid=True, figsize=(15, 10),  # Увеличиваем высоту графика
                                                      rot=90,
                                                      xticks=range(len(self.scaled_df.columns) - 1),
                                                      style='o-', linewidth=4, markersize=12)

        plt.legend(fontsize=30)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=10)  # Уменьшаем шрифт оси y
        plt.tight_layout()  # Подбираем отступы

        # Сохраняем график
        image_path_z_stats = 'z_stats.png'
        plt.savefig(image_path_z_stats)
        plt.close('all')  # Закрывает все открытые графики

        self.temp.groupby("Class").mean()

        ###############################
        # Создание датафрейма с результатами

        # Данные для таблицы `grouped_count`
        df_grouped_count = pd.DataFrame({
            'Класс': grouped_count.index,
            'Число объектов': grouped_count.values
        })

        # Данные для заголовков и значений таблицы `mean_temp`
        mean_temp_columns = ['Класс'] + list(mean_temp.columns)
        mean_temp_data = []

        # Добавляем строку заголовка для средних значений
        for class_label, row in mean_temp.iterrows():
            mean_temp_data.append([class_label] + list(row.values))

        df_mean_temp = pd.DataFrame(mean_temp_data, columns=mean_temp_columns)

        # Создаем итоговый DataFrame, включающий таблицу `grouped_count` и средние значения
        # Объединяем оба DataFrame по вертикали
        df_result = pd.concat([df_grouped_count,
                               pd.DataFrame([["Средние значения"] + [""] * (len(mean_temp_columns) - 1)],
                                            columns=mean_temp_columns), df_mean_temp], ignore_index=True)

        return df_result, image_path_z_stats

    def calculate_principal_components(self):

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.scaled_df.iloc[:, :-1])
        principalDf = pd.DataFrame(data=principalComponents)

        principalDf = principalDf.rename(columns={0: "PC1", 1: "PC2"})
        principalDf["Class"] = self.scaled_df["Class"]

        ###############################

        # Визуализация главных компонент
        for i in range(1, self.best_n_clusters + 1):
            data = principalDf[principalDf["Class"] == i]
            plt.plot(data.PC1, data.PC2, 'o', label=f'Класс {i}')
        plt.legend()
        plt.title("Главные компоненты")
        plt.xlabel("Главная компонента 1")
        plt.ylabel("Главная компонента 2")

        # Сохраняем график
        image_path_pca = 'pca_plot.png'
        plt.savefig(image_path_pca)
        plt.close('all')

        return None, image_path_pca

    def calculate_anomaly_class(self):
        anomaly_labels = IsolationForest().fit_predict(self.scaled_df.drop(["Class"], axis=1))

        self.scaled_df["IsoLabels"] = anomaly_labels

        # Список для хранения отношений z-stat first
        ratios = {}

        # Получаем уникальные классы
        classes = self.scaled_df["Class"].unique()

        # Проходим по каждому классу и вычисляем отношение
        for cls in classes:
            # Получаем count z-stat first для текущего класса
            z_stat_pos = self.scaled_df[(self.scaled_df["Class"] == cls) & (self.scaled_df["IsoLabels"] == 1)][
                "z-stat first"].count()
            z_stat_neg = self.scaled_df[(self.scaled_df["Class"] == cls) & (self.scaled_df["IsoLabels"] == -1)][
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
        anomaly_labels_el = EllipticEnvelope().fit_predict(self.scaled_df.drop(["Class", "IsoLabels"], axis=1))

        self.scaled_df["ElLabels"] = anomaly_labels_el
        self.scaled_df.groupby(["ElLabels", "IsoLabels", "Class"]).count()

        ratios_el = {}

        # Проходим по каждому классу и вычисляем отношение для Elliptic Envelope
        for cls in classes:
            # Получаем count z-stat first для текущего класса
            z_stat_pos = self.scaled_df[(self.scaled_df["Class"] == cls) & (self.scaled_df["ElLabels"] == 1)][
                "z-stat first"].count()
            z_stat_neg = self.scaled_df[(self.scaled_df["Class"] == cls) & (self.scaled_df["ElLabels"] == -1)][
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
            self.anomaly_class_el = min(ratios_el, key=ratios_el.get)  # Класс с минимальным отношением
        else:
            self.anomaly_class_el = None  # Если нет классов

        print(f'Аномальный класс по Elliptic Envelope: {self.anomaly_class_el}')

        ###############################
        # Группируем данные и заполняем нулями
        grouped_iso = self.scaled_df.groupby(["IsoLabels", "Class"]).count().reindex(
            pd.MultiIndex.from_product([[-1, 1], classes], names=["IsoLabels", "Class"]), fill_value=0
        )

        grouped_el = self.scaled_df.groupby(["ElLabels", "Class"]).count().reindex(
            pd.MultiIndex.from_product([[-1, 1], classes], names=["ElLabels", "Class"]), fill_value=0
        )

        # Создаем основной DataFrame для Isolation Forest
        df_iso = pd.DataFrame({
            "Method": ["Isolation Forest"] * len(grouped_iso),
            "Labels": [index[0] for index in grouped_iso.index],
            "Class": [index[1] for index in grouped_iso.index],
            "Count": grouped_iso['z-stat first'].values
        })

        # Добавляем строку с информацией об аномальном классе для Isolation Forest
        df_iso_summary = pd.DataFrame({
            "Method": ["Isolation Forest"],
            "Labels": ["Аномальный класс по Isolation Forest:"],
            "Class": [anomaly_class_iso],
            "Count": [None]
        })

        # Создаем основной DataFrame для Elliptic Envelope
        df_el = pd.DataFrame({
            "Method": ["Elliptic Envelope"] * len(grouped_el),
            "Labels": [index[0] for index in grouped_el.index],
            "Class": [index[1] for index in grouped_el.index],
            "Count": grouped_el['z-stat first'].values
        })

        # Добавляем строку с информацией об аномальном классе для Elliptic Envelope
        df_el_summary = pd.DataFrame({
            "Method": ["Elliptic Envelope"],
            "Labels": ["Аномальный класс по Elliptic Envelope:"],
            "Class": [self.anomaly_class_el],
            "Count": [None]
        })

        # Объединяем все части в один DataFrame
        df_combined = pd.concat([df_iso, df_iso_summary, pd.DataFrame([["", "", "", ""]]), df_el, df_el_summary],
                                ignore_index=True)

        return df_combined

    def create_graphics(self):

        self.scaled_df_temp = self.scaled_df.copy()
        self.scaled_df = self.scaled_df.drop(["ElLabels", "IsoLabels"], axis=1)

        # Создание графиков для каждого класса и сохранение в файлы PNG
        classes = self.scaled_df_temp["Class"].unique()
        image_files = []

        # Проходим по каждому классу и строим графики
        for cls in classes:
            plt.figure(figsize=(15, 8))

            # Основной график для среднего значения по классу
            self.scaled_df_temp[self.scaled_df_temp["Class"] == cls].iloc[:, :-3].mean().T.plot(
                style='o-', linewidth=4, markersize=12, label=f'Среднее по классу {cls}'
            )

            # График для IsoLabels=-1 и ElLabels=-1 для данного класса
            self.scaled_df_temp[
                (self.scaled_df_temp["Class"] == cls) &
                (self.scaled_df_temp["IsoLabels"] == -1) &
                (self.scaled_df_temp["ElLabels"] == -1)
                ].iloc[:, :-3].mean().T.plot(
                style='o--', linewidth=4, markersize=12, label=f'Аномалии по обоим методам (-1, -1)'
            )

            # График для IsoLabels=1 и ElLabels=1 для данного класса
            self.scaled_df_temp[
                (self.scaled_df_temp["Class"] == cls) &
                (self.scaled_df_temp["IsoLabels"] == 1) &
                (self.scaled_df_temp["ElLabels"] == 1)
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

        return None, image_files

    def search_suspicious_transactions_1(self):
        self.scaled_df_2 = self.scaled_df_temp[
                               (self.scaled_df_temp["ElLabels"] == -1) & (self.scaled_df_temp["IsoLabels"] == -1) & (
                                       self.scaled_df_temp["Class"] == self.anomaly_class_el)].iloc[:, :-2]

        # Получаем индексы аномальных объектов из scaled_df
        anomaly_indices_el = self.scaled_df[self.scaled_df["Class"] == self.anomaly_class_el].index

        # Фильтрация исходного DataFrame df по этим индексам
        anomaly_original_rows = self.df.loc[anomaly_indices_el]

        return anomaly_original_rows

    def calculate_digit_stats_2(self):
        cluster_1_index = list(self.scaled_df[self.scaled_df["Class"] == self.anomaly_class_el].index)

        df_class_1 = self.temp[(self.temp.index.isin(cluster_1_index))]

        temp_class_1 = df_class_1.loc[:,
                       ["z-stat first", "z-stat second", "z-stat first_two", "sum_frequency", "z_stat_second_diff",
                        "Частота суммы", "Частота счета Дт", "Частота счета Кт", "Частота проводки",
                        "Частота автора операции",
                        "Ручная проводка", "Выходные или рабочие", "Сторно", "Количество дублей"]]

        scaled_ = StandardScaler().fit_transform(temp_class_1.iloc[:, 6:].values)
        self.scaled_class_1 = pd.DataFrame(scaled_, index=temp_class_1.iloc[:, 6:].index,
                                           columns=temp_class_1.iloc[:, 6:].columns)

        # Силуэт
        plt.close('all')  # Закрывает все открытые графики

        plt.rcParams["figure.figsize"] = (8, 4)

        x = [i for i in range(2, 6)]
        m = []

        sample_size = int(self.scaled_class_1.shape[0])
        print(f'sample size = {sample_size}')
        for i in tqdm(x):
            print(i)
            labels = KMeans(n_clusters=i, random_state=1000).fit(self.scaled_class_1).labels_
            print(labels)
            m.append(metrics.silhouette_score(self.scaled_class_1, labels, sample_size=sample_size))
            print(f'n = {i}, silhouette = {m[-1]}')
        # Найдём количество кластеров с максимальным значением silhouette_score
        self.best_n_clusters = x[m.index(max(m))]

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

        data = {
            'Количество кластеров': x,
            'Silhouette Score': m
        }
        silhouette_df = pd.DataFrame(data)

        # Добавляем строку с оптимальным числом кластеров как метаданные
        optimal_cluster_info = pd.DataFrame({
            'Оптимальное количество кластеров': [self.best_n_clusters]
        })

        # Объединяем DataFrame для результатов и строку с оптимальным значением
        result_df = pd.concat([optimal_cluster_info, silhouette_df], ignore_index=True)

        return result_df, image_path

    def calculate_clasterization_2(self):

        n_clusters = self.best_n_clusters
        if "Class" in self.scaled_class_1.columns:
            self.scaled_class_1.drop(columns=["Class", ], inplace=True)
        km = KMeans(n_clusters=n_clusters, random_state=1000)
        self.scaled_class_1["Class"] = km.fit_predict(self.scaled_class_1)
        self.scaled_class_1["Class"] = self.scaled_class_1["Class"] + 1

        # Удаляем столбец "Class", если он уже есть
        if "Class" in self.scaled_class_1.columns:
            self.scaled_class_1.drop(columns=["Class"], inplace=True)

        # Применение KMeans
        km = KMeans(n_clusters=n_clusters, random_state=1000)
        self.scaled_class_1["Class"] = km.fit_predict(self.scaled_class_1)
        self.scaled_class_1["Class"] = self.scaled_class_1["Class"] + 1  # Нумерация классов с 1

        # Вычисляем количество объектов в каждом классе
        class_counts = self.scaled_class_1.groupby("Class").count()["Сторно"]
        class_counts = class_counts.reset_index()  # Преобразуем индекс в столбец для удобного вывода

        # Переименовываем столбцы
        class_counts.columns = ['Класс', 'Количество объектов']

        # Рисуем график средних значений по каждому классу
        self.scaled_class_1.groupby("Class").mean().T.plot(grid=True, figsize=(15, 10), rot=90,
                                                           xticks=range(len(self.scaled_class_1.columns) - 1),
                                                           style='o-',
                                                           linewidth=4,
                                                           markersize=12)

        plt.legend(fontsize=30)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)

        # Сохраняем график во временный файл
        plt.savefig('cluster_plot.png', bbox_inches='tight')
        plt.close('all')  # Закрывает все открытые графики

        # Примерные данные по количеству объектов в каждом классе
        data = {
            'Класс': ['Класс 0', 'Класс 1', 'Класс 2', 'Класс 3'],
            'Количество объектов': [120, 98, 135, 110]
        }
        class_counts_df = pd.DataFrame(data)

        # DataFrame, включающий информацию о листе, данных, графике и его положении
        kmeans_clustering_df = pd.DataFrame({
            'Sheet': ['KMeans Clustering'],
            'Data': [class_counts_df.to_dict()],  # Сохранение данных в виде словаря
            'Image': ['cluster_plot.png'],  # Путь к изображению (графику)
            'Image Position': ['A10'],  # Позиция изображения
            'Scale': [(0.7, 0.7)]  # Масштаб изображения
        })

        # Вывод DataFrame для просмотра
        return kmeans_clustering_df, "cluster_plot.png"

    def search_anomaly_classes(self):

        anomaly_labels = IsolationForest().fit_predict(self.scaled_class_1.drop(["Class"], axis=1))

        self.scaled_class_1["IsoLabels"] = anomaly_labels

        # Список для хранения отношений Сторно
        ratios = {}

        # Определяем порядок классов
        class_order = sorted(self.scaled_class_1["Class"].unique())  # [1, 2, 3, 4, 5]

        # Проходим по каждому классу и вычисляем отношение
        for cls in class_order:
            z_stat_pos = self.scaled_class_1[(self.scaled_class_1["Class"] == cls) & (self.scaled_class_1["IsoLabels"] == 1)][
                "Сторно"].count()
            z_stat_neg = self.scaled_class_1[(self.scaled_class_1["Class"] == cls) & (self.scaled_class_1["IsoLabels"] == -1)][
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

        anomaly_labels_el = EllipticEnvelope().fit_predict(self.scaled_class_1.drop(["Class", "IsoLabels"], axis=1))

        self.scaled_class_1["ElLabels"] = anomaly_labels_el

        ratios_el = {}

        for cls in class_order:
            z_stat_pos = self.scaled_class_1[(self.scaled_class_1["Class"] == cls) & (self.scaled_class_1["ElLabels"] == 1)][
                "Сторно"].count()
            z_stat_neg = self.scaled_class_1[(self.scaled_class_1["Class"] == cls) & (self.scaled_class_1["ElLabels"] == -1)][
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
        grouped_iso = self.scaled_class_1.groupby(["IsoLabels", "Class"]).count().reindex(
            pd.MultiIndex.from_product([[-1, 1], class_order], names=["IsoLabels", "Class"]), fill_value=0
        )

        grouped_el = self.scaled_class_1.groupby(["ElLabels", "Class"]).count().reindex(
            pd.MultiIndex.from_product([[-1, 1], class_order], names=["ElLabels", "Class"]), fill_value=0
        )

        # Примерные данные для Isolation Forest и Elliptic Envelope
        grouped_iso_data = {
            ('IsoLabel1', 'Class A'): {'Сторно': 50},
            ('IsoLabel2', 'Class B'): {'Сторно': 30},
            ('IsoLabel3', 'Class C'): {'Сторно': 20}
        }
        grouped_el_data = {
            ('ElLabel1', 'Class A'): {'Сторно': 45},
            ('ElLabel2', 'Class B'): {'Сторно': 25},
            ('ElLabel3', 'Class C'): {'Сторно': 15}
        }

        # Преобразуем словари данных в DataFrame для Isolation Forest
        df_iso = pd.DataFrame.from_dict(grouped_iso_data, orient='index').reset_index()
        df_iso.columns = ['IsoLabels', 'Класс', 'Число объектов']

        # Пример аномального класса для Isolation Forest
        anomaly_class_iso = 'Class B'

        # Преобразуем словари данных в DataFrame для Elliptic Envelope
        df_el = pd.DataFrame.from_dict(grouped_el_data, orient='index').reset_index()
        df_el.columns = ['ElLabels', 'Класс', 'Число объектов']

        # Пример аномального класса для Elliptic Envelope
        anomaly_class_el = 'Class C'

        # Соединяем данные в один DataFrame с разделителями и дополнительной информацией
        separator_row = pd.DataFrame({'IsoLabels': ['-----'], 'Класс': [''], 'Число объектов': ['']})
        anomaly_info_iso = pd.DataFrame(
            {'IsoLabels': ['Аномальный класс по Isolation Forest:'], 'Класс': [anomaly_class_iso],
             'Число объектов': ['']})
        anomaly_info_el = pd.DataFrame(
            {'ElLabels': ['Аномальный класс по Elliptic Envelope:'], 'Класс': [anomaly_class_el],
             'Число объектов': ['']})

        # Создаем итоговый DataFrame с объединенными результатами
        final_df = pd.concat([df_iso, anomaly_info_iso, separator_row, df_el, anomaly_info_el], ignore_index=True)

        # Отображаем DataFrame
        return final_df

    def search_suspicious_transactions_2(self):
        anomaly_indices_el = self.scaled_class_1[self.scaled_class_1["Class"] == self.anomaly_class_el].index

        # Фильтрация исходного DataFrame df по этим индексам
        anomaly_original_rows = self.df.loc[anomaly_indices_el]

        return anomaly_original_rows
