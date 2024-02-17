from flask import Flask, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = 'uploaded_image.png'
    file.save(filename)

    # Загрузка исходного изображения и изображения для сравнения
    img1 = cv2.imread('DALL3.png', cv2.IMREAD_GRAYSCALE)  # Исходное изображение
    img2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Загруженное изображение

    # Инициализация SIFT детектора
    sift = cv2.SIFT_create()

    # Нахождение ключевых точек и дескрипторов для обоих изображений
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Инициализация BFMatcher
    bf = cv2.BFMatcher()

    # Сопоставление дескрипторов с использованием BFMatcher и KNN
    matches = bf.knnMatch(des1, des2, k=2)

    # Отсеивание ложных совпадений с использованием теста отношения Лоу
    good = []
    for m, n in matches:
        if m.distance < 0.35 * n.distance:
            good.append([m])

    # Проверка наличия совпадений
    if len(good) > MIN_MATCH_COUNT:
        return 'Успешно', 200
    else:
        return 'Неудачно', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)