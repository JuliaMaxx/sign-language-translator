# app.py
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("models/asl_model.h5")

@app.route("/")
def index():
    return "ASL Translator API is running."

if __name__ == "__main__":
    app.run()
