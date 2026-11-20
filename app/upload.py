import sqlite3
import uuid
from pathlib import Path
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
DATABASE_PATH = BASE_DIR / "data" / "uploads.db"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]


def create_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                class_name TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_images(images, class_name):
    if class_name not in class_names:
        raise ValueError("Invalid class name.")
    if not images or len(images) > 20 or any(not image.filename for image in images):
        raise ValueError("Please upload between 1 and 20 images.")

    class_folder = UPLOAD_FOLDER / class_name
    class_folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            for uploaded in images:
                extension = Path(uploaded.filename).suffix.lower()
                if extension not in (".jpg", ".jpeg", ".png"):
                    raise ValueError("Only JPG and PNG images are allowed.")

                try:
                    uploaded.stream.seek(0)
                    with Image.open(uploaded.stream) as image:
                        if image.format not in ("JPEG", "PNG"):
                            raise ValueError("The file is not a JPG or PNG image.")
                        image.verify()
                    uploaded.stream.seek(0)
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    raise ValueError("One of the uploaded images is invalid.") from error

                filename = f"{uuid.uuid4()}{extension}"
                destination = class_folder / filename
                uploaded.save(destination)
                saved_paths.append(destination)
                connection.execute(
                    "INSERT INTO uploaded_images (filename, class_name) VALUES (?, ?)",
                    (filename, class_name)
                )
    except Exception:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise

    return len(saved_paths)


create_database()
