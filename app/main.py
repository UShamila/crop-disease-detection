from flask import Flask, request, jsonify, render_template
from app.prediction import predict_image
from app.upload import save_images, class_names
from app.retraining import start_retraining, get_status
import time
from app.monitoring import record_prediction, get_monitoring
from app.security import admin_required


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Display the home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# Check whether the application is running
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "message": "Crop Disease Detection API is working"
    })


# Predict the disease from an uploaded image
@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()

    if "image" not in request.files or not request.files["image"].filename:
        record_prediction(False, (time.perf_counter() - start) * 1000)
        return jsonify({"error": "Please upload an image"}), 400

    try:
        result = predict_image(request.files["image"])

        response_ms = (time.perf_counter() - start) * 1000
        record_prediction(True, response_ms)

        return jsonify(result)

    except Exception:
        response_ms = (time.perf_counter() - start) * 1000
        record_prediction(False, response_ms)

        return jsonify({
            "error": "Prediction failed. Please check the image."
        }), 400

@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": class_names})


@app.route("/upload", methods=["POST"])
@admin_required
def upload_images():
    if "images" not in request.files:
        return jsonify({"error": "Please select images."}), 400

    class_name = request.form.get("class_name")
    images = request.files.getlist("images")

    try:
        count = save_images(images, class_name)

        return jsonify({
            "message": "Images uploaded successfully.",
            "uploaded_count": count,
            "class_name": class_name
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/retrain", methods=["POST"])
@admin_required
def retrain():
    if not start_retraining():
        return jsonify({
            "error": "Retraining is already running."
        }), 409

    return jsonify({
        "message": "Retraining started."
    }), 202


@app.route("/retrain/status", methods=["GET"])
def retrain_status():
    return jsonify(get_status())

@app.route("/monitoring", methods=["GET"])
def monitoring():
    return jsonify(get_monitoring())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)