# app.py
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route("/")
def index():
    return "ASL Translator API is running."

if __name__ == "__main__":
    app.run()
