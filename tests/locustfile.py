from locust import HttpUser, task, between
from pathlib import Path


IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "uploads" / "Apple___Cedar_apple_rust" / "0bb1e6ff-2b88-4bc5-8cd4-fd131e6cbbe2.png"


class CropDiseaseUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_disease(self):
        with open(IMAGE_PATH, "rb") as image:
            self.client.post(
                "/predict",
                files={"image": ("leaf.png", image, "image/png")},
                name="/predict",
                timeout=120
            )