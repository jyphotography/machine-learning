import json
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
import urllib.request
from PIL import Image
from io import BytesIO

def download_image(url):
    response = urllib.request.urlopen(url)
    buffer = response.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0


interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))

    x = np.array(img, dtype="float32")
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event["url"]
    pred = predict(url)
    result = {"prediction": pred}

    return result