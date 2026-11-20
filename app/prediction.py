from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]


def predict_image(image_file):
    try:
        with Image.open(image_file.stream) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPG and PNG images are allowed.")
            image = image.convert("RGB")
            image = image.resize((128, 128), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Please upload a valid JPG or PNG image.") from error

    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }
