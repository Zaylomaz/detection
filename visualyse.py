from flask import Flask, request, send_from_directory
import cv2
import numpy as np
import os
import datetime
from flask import jsonify
from multiprocessing import Process

MIN_MATCH_COUNT = 10  # Define the variable before using it
app = Flask(__name__)

import time
import shutil
def recognize(filename, template):
    # Проверка существования файла
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден")
        return 'Неудачно', 400

    # Добавление небольшой задержки перед чтением файла
    time.sleep(1)

    # Загрузка исходного изображения и изображения для сравнения
    img1 = cv2.imread(template, 0)  # Исходное изображение
    img2 = cv2.imread(filename, 0)  # Загруженное изображение

    # Проверка корректности изображения
    if img2 is None or img2.dtype != np.uint8:
        print(f"Изображение {filename} пустое или имеет некорректную глубину")
        return 'Неудачно', 400

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
        # Создание директории с текущей датой, если она еще не существует
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        new_folder = os.path.join('recognized_images', current_date)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        # Перемещение файла в новую директорию
        new_path = os.path.join(new_folder, os.path.basename(filename))
        shutil.move(filename, new_path)
        print(f"Файл {filename} успешно перемещен в {new_path}")

        # Визуализация совпадений
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        # Сохранение визуализации
        visualization_path = os.path.join(new_folder, 'visualization_' + os.path.basename(filename))
        cv2.imwrite(visualization_path, img3)
        print(f"Визуализация сохранена в {visualization_path}")

        return 'Успешно', 200
    else:
        print(f"Совпадений для файла {filename} не найдено")
        return 'Неудачно', 400
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет файла изображения в запросе'}), 400

    file = request.files['image']
    if file.filename == '':
        return 400

    if file:
        # Преобразование файла изображения в массив numpy
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)  # Используйте np.frombuffer вместо np.fromstring
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Сохранение изображения
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.jpg"
        output_folder = 'uploaded_images'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path, img)

        # Запуск функции recognize в отдельном процессе
        p = Process(target=recognize, args=(output_path,"C:\dev\detection\Template (1).jpeg"))
        p.start()
        p = Process(target=recognize, args=(output_path,"C:\dev\detection\Template (2).jpeg"))
        p.start()

        return 200
def recognize_all():
        for filename in os.listdir('uploaded_images'):
            recognize('uploaded_images' + '/' + filename,"C:\dev\detection\Template (1).jpeg")
            time.sleep(1)
            recognize('uploaded_images' + '/' + filename,"C:\dev\detection\Template (2).jpeg")

if __name__ == '__main__':

    p = Process(target=recognize_all)
    p.start()
    # recognize_all()
    app.run(host='0.0.0.0', port=5000)