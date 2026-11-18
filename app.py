"""Flask web app that predicts the facial expression in an uploaded image.

Routes:
    GET  /         Upload page.
    POST /predict  Multipart form with a ``file`` image. Returns
                   ``{"result": "happy", "confidence": 0.82, "probabilities": {...}}``.

Run:
    python app.py
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fer_cnn.h5"

# Output order of the network. Keras' flow_from_directory assigns class
# indexes alphabetically by folder name, which gives this order.
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

INPUT_SIZE = (48, 48)  # FER-2013 image size the model was trained on

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    """Turn any image into the (1, 48, 48, 3) batch the model expects.

    The training images were 48x48 grayscale faces, loaded as 3-channel RGB by
    Keras. We reproduce that: grayscale -> resize -> copy into 3 channels ->
    scale pixels to [0, 1].
    """
    gray = np.asarray(image.convert("L").resize(INPUT_SIZE), dtype=np.float32)
    rgb = np.stack([gray] * 3, axis=-1) / 255.0
    return np.expand_dims(rgb, axis=0)  # add the batch dimension


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    try:
        image = Image.open(file.stream)
    except UnidentifiedImageError:
        return jsonify({"error": "The file is not a valid image"}), 400

    probs = model.predict(preprocess(image), verbose=0)[0]
    best = int(np.argmax(probs))

    return jsonify({
        "result": EMOTIONS[best],
        "confidence": float(probs[best]),
        "probabilities": {emotion: float(p) for emotion, p in zip(EMOTIONS, probs, strict=True)},
    })


if __name__ == "__main__":
    app.run(debug=True)
