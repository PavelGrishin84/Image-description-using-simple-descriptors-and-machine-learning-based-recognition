"""Описание изображений простыми дескрипторами и распознавание объектов
на основе сравнения с эталоном (Python-порт скрипта
Image_description_and_ML_recognition.m).

Наборы данных:
- изображения букв английского алфавита notMNIST (kaggle.com);
- изображения одежды Fashion MNIST PNG (kaggle.com).

Требуемые библиотеки: numpy, scikit-learn, Pillow, matplotlib.
"""

import os
import time

import numpy as np
from PIL import Image
from scipy.stats import mode as sp_mode
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Путь к папке с изображениями (внутри — подпапки с именами классов),
DATA_DIR = "C:/Users/gps.84/Downloads/archive/train"

IMG_SIZE = 28  # ожидаемый размер изображений (28x28)


# 1. Загрузка изображений (анал imageDatastore)
def load_dataset(data_dir):
    """Собирает пути файлов и метки классов из имён подпапок."""
    files, labels = [], []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                files.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    return files, np.array(labels, dtype=object)


# 2. Вычисление дескрипторов
def compute_descriptors(files, labels):
    """Вычисляет 4 варианта дескрипторов для каждого изображения.
    D_1: [X Y'] — сумма по столбцам и по строкам (2*28 признаков)
    D_2: X - Y' — разность проекций (28 признаков)
    D_3: abs(X - Y') — модуль разности проекций (28 признаков)
    D_4: mean/std/mode/median от D_1 (4 признака)
    """
    n = len(files)
    d1 = np.zeros((n, 2 * IMG_SIZE), dtype=np.float32)
    d2 = np.zeros((n, IMG_SIZE), dtype=np.float32)
    d3 = np.zeros((n, IMG_SIZE), dtype=np.float32)
    d4 = np.zeros((n, 4), dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    t0 = time.time()  # ана tic ... toc
    for i, path in enumerate(files):
        try:
            # Загрузка изображения из файла
            img = Image.open(path)

            # Пороговая бинаризация изображения:
            # порог Отсу на нормализованном ярком изображении)
            gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
            thr = _otsu_threshold(gray)
            binary = (gray > thr).astype(np.uint8)

            # Вычисление дескрипторов
            x = binary.sum(axis=0, dtype=np.float64)  # сумма значений пикселей по столбцам
            y = binary.sum(axis=1, dtype=np.float64)  # сумма значений пикселей по строкам

            row = np.concatenate([x, y])              # вариант дескриптора № 1
            d1[i, :] = row
            d2[i, :] = x - y                          # вариант дескриптора № 2
            d3[i, :] = np.abs(x - y)                  # вариант дескриптора № 3
            d4[i, :] = [                              # вариант дескриптора № 4
                row.mean(),
                row.std(ddof=1),                      # аналогично std MATLAB (N-1)
                sp_mode(row, keepdims=False).mode,    # мода (при нескольких — наименьшее значение)
                np.median(row),
            ]
        except Exception:
            print("Ошибка чтения файла!")
            valid[i] = False
    elapsed = time.time() - t0
    print(f"Время вычисления дескрипторов: {elapsed:.3f} c")

    # Отбрасываем файлы, которые не удалось прочитать
    return d1[valid], d2[valid], d3[valid], d4[valid], labels[valid]


def _otsu_threshold(gray):
    """Порог Отсу (как по умолчанию в imbinarize 'global')."""
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.clip(omega * (1 - omega), 1e-12, None)
    return int(np.argmax(sigma_b2)) / 255.0


# Обучение моделей
def train_models(d1, labels, class_names):
    """Обучает 4 модели, соответствующие моделям из MATLAB-скрипта."""
    models = {}

    # Обучение ансамбля K - ближайших соседей (KNN)
    models["Ансамбль ближайших соседей (KNN)"] = SubspaceKnnEnsemble(
        n_estimators=30, max_features=28, random_state=1
    ).fit(d1, labels)

    # Обучение ансамбля деревьев решений (Bagging)
    models["Ансамбль деревьев решений (TREE)"] = BaggingClassifier(
        n_estimators=30, random_state=1
    ).fit(d1, labels)

    # Обучение полносвязной нейронной сети
    models["Полносвязная нейронная сеть (NeuralNetwork)"] = StandardizedModel(
        MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            alpha=0.0,                 # Lambda = 0 (без регуляризации)
            max_iter=1000,             # IterationLimit
            early_stopping=False,
            random_state=1,
        )
    ).fit(d1, labels)

    # Обучение машин опорных векторов (SVM, ECOC one-vs-one, полиномиальное ядро 2-й степени)
    models["Машины опорных векторов классификации (SVC)"] = StandardizedModel(
        OneVsOneSVCPoly2(degree=2, C=1.0)
    ).fit(d1, labels, class_names)

    return models


class StandardizedModel:
    """Обёртка, применяющая стандартизацию (по параметрам обучающей выборки)
    перед обучением/предсказанием внутренней модели — аналог 'Standardize', true."""

    def __init__(self, model):
        self.model = model

    def fit(self, x, y, *args, **kwargs):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        self.model.fit((x - self.mean_) / self.std_, y, *args, **kwargs)
        return self

    def predict(self, x):
        return self.model.predict((x - self.mean_) / self.std_)


class SubspaceKnnEnsemble:
    """Ансамбль KNN методом Subspace (случайные подпространства признаков),
    """

    def __init__(self, n_estimators=30, max_features=28, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, x, y):
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.n_features_ = x.shape[1]
        self.idx_ = [
            rng.choice(self.n_features_,
                       size=min(self.max_features, self.n_features_),
                       replace=False)
            for _ in range(self.n_estimators)
        ]
        self.estimators_ = [
            _SimpleKNN().fit(x[:, idx], y) for idx in self.idx_
        ]
        return self

    def predict(self, x):
        votes = np.zeros((x.shape[0], len(self.classes_)), dtype=float)
        for est, idx in zip(self.estimators_, self.idx_):
            pred = est.predict(x[:, idx])
            for j, cls in enumerate(self.classes_):
                votes[:, j] += (pred == cls)
        return self.classes_[votes.argmax(axis=1)]


class _SimpleKNN:
    """Базовый KNN (k=5, равномерное голосование) """

    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.x_ = x
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, x):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k).fit(self.x_)
        _, neigh = nn.kneighbors(x)
        neighbor_labels = self.y_[neigh]
        out = np.empty(x.shape[0], dtype=object)
        for i, lab in enumerate(neighbor_labels):
            vals, counts = np.unique(lab, return_counts=True)
            out[i] = vals[counts.argmax()]
        return out


class OneVsOneSVCPoly2:
    """ ECOC one-vs-one из SVM с полиномиальным ядром 2-й степени,
    """

    def __init__(self, degree=2, C=1.0):
        self.degree = degree
        self.C = C

    def fit(self, x, y, class_names=None):
        self.classes_ = np.array(class_names if class_names is not None
                                 else np.unique(y), dtype=object)
        self.pairwise_ = []
        for a in range(len(self.classes_)):
            for b in range(a + 1, len(self.classes_)):
                mask = np.isin(y, [self.classes_[a], self.classes_[b]])
                svc = SVC(kernel="poly", degree=self.degree, C=self.C)
                svc.fit(x[mask], y[mask])
                self.pairwise_.append((a, b, svc))
        return self

    def predict(self, x):
        votes = np.zeros((x.shape[0], len(self.classes_)))
        for a, b, svc in self.pairwise_:
            pred = svc.predict(x)
            votes[:, a] += (pred == self.classes_[a])
            votes[:, b] += (pred == self.classes_[b])
        return self.classes_[votes.argmax(axis=1)]


# Распознавание тестовых изображений и оценка точности классификации
def evaluate(models, descriptors, labels):
    """Вывод диаграммы ошибок и усреднённой точности для каждой модели
    """
    import matplotlib.pyplot as plt

    classes = np.unique(labels)
    for name, model in models.items():
        pred = model.predict(descriptors)
        cm_raw = confusion_matrix(labels, pred, labels=classes)  # строки — эталон, столбцы — прогноз

        # 1. Считаем accuracy по ИСХОДНОЙ (целочисленной) матрице
        acc = np.trace(cm_raw) / cm_raw.sum()
        print(f"Усредненное значение точности классификации изображений {name}: {acc:f}")

        # 2. Нормализуем КОПИЮ матрицы для визуализации (по строкам, от 0 до 1)
        cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # защита от NaN

        fig, ax = plt.subplots()
        im = ax.imshow(cm_norm.T, cmap="Blues", origin="upper")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Доля (от 0 до 1)")  # теперь это доля, а не число изображений

        # 3. Порог для цвета текста — от нормализованной матрицы
        thresh = cm_norm.max() / 2

        # 4. Подписываем ячейки — теперь используем формат ".2f" для float
        for i in range(len(classes)):
            for j in range(len(classes)):
                # Вариант А: вывод в виде десятичной дроби (например, 0.85)
                ax.text(j, i, format(cm_norm[i, j], ".2f"),
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")

                # Вариант Б (если хотите в процентах, раскомментируйте и удалите вариант А):
                # ax.text(j, i, format(cm_norm[i, j], ".0%"),
                #         ha="center", va="center",
                #         color="white" if cm_norm[i, j] > thresh else "black")

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)  # прогноз (столбцы)
        ax.set_yticklabels(classes)  # эталон (строки)
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Эталон")
        ax.set_title(name)
    plt.show()


# %% Основная программа
if __name__ == "__main__":
    # 1. Загрузка списка изображений
    files, labels = load_dataset(DATA_DIR)
    print(f"Загружено изображений: {len(files)}")

    # 2. Вычисление дескрипторов D_1 .. D_4
    d1, d2, d3, d4, labels = compute_descriptors(files, labels)

    # Названия классов набора (анал classNames; при необходимости задайте вручную,
    # например {'0';'1';...;'9'} для цифр notMNIST)
    class_names = np.unique(labels)

    # Разделение на обучающую и тестовую выборки (75/25, стратификация по меткам)
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=1
    )

    # 3. Обучение моделей на тренировочных данных
    models = train_models(d1[idx_train], labels[idx_train], class_names)

    # Выбор дескриптора (D_1, D_2, D_3 или D_4)
    DESCRIPTORS = {"D_1": d1, "D_2": d2, "D_3": d3, "D_4": d4}["D_1"]

    # 4. Распознавание тестовых изображений и оценка точности
    evaluate(models, DESCRIPTORS[idx_test], labels[idx_test])
