from flask import Flask, request, jsonify
import os
import datetime
import cv2 as cv
import numpy as np

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет файла изображения в запросе'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Нет выбранного файла'}), 400

    if file:
        # Преобразование файла изображения в массив numpy
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv.imdecode(npimg, cv.IMREAD_COLOR)

        # Сохранение изображения
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.jpg"
        output_folder = 'uploaded_images'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, filename)

        cv.imwrite(output_path, img)

        return jsonify({'message': f'Изображение успешно загружено и сохранено как {filename}'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
