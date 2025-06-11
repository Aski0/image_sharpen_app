from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model1000x512_b1_ep30.h5")


def sharpen_image_ai(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = img.astype('float32')

    input_tensor = np.expand_dims(img, axis=0)  # dodaj batch
    output = model.predict(input_tensor)[0]
    output = np.clip(output, 0, 255).astype('uint8')

    is_success, buffer = cv2.imencode(".jpg", output)
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    processed_bytes = sharpen_image_ai(image_file.read())

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(processed_bytes)
    temp.close()

    return send_file(temp.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
