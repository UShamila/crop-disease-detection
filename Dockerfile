FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY models/class_names.txt ./models/class_names.txt

EXPOSE 5000

CMD ["python", "-m", "app.main"]
