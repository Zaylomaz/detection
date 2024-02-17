import cv2
import matplotlib.pyplot as plt
import numpy as np

# Загрузка исходного изображения и изображения для сравнения
img1 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # Исходное изображение
images =  ['image2.jpg'] # Пути к изображениям в массиве

# Инициализация SIFT детектора
sift = cv2.SIFT_create()

# Нахождение ключевых точек и дескрипторов исходного изображения
kp1, des1 = sift.detectAndCompute(img1, None)

# Инициализация BFMatcher
bf = cv2.BFMatcher()

MIN_MATCH_COUNT = 10  # Define the variable before using it

for img_path in images:
    img2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Загрузка изображения из массива
    kp2, des2 = sift.detectAndCompute(img2, None)  # Ключевые точки и дескрипторы изображения из массива

    # Сопоставление дескрипторов с использованием BFMatcher
    matches = bf.knnMatch(des1, des2, k=2)

    # Отсеивание ложных совпадений с использованием теста отношения Лоу
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Проверка наличия объекта по количеству хороших совпадений
    if len(good) > MIN_MATCH_COUNT:
        print(f"Объект найден на изображении {img_path} с {len(good)} хорошими совпадениями.")
    else:
        print(f"Объект не найден на изображении {img_path}.")