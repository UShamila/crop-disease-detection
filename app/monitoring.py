import time
from datetime import datetime
from threading import Lock

started_at = time.monotonic()
lock = Lock()
successful_predictions = 0
failed_predictions = 0
total_response_ms = 0.0
last_prediction = None


def record_prediction(success, response_ms):
    global successful_predictions, failed_predictions, total_response_ms, last_prediction
    with lock:
        if success:
            successful_predictions += 1
            total_response_ms += response_ms
        else:
            failed_predictions += 1
        last_prediction = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_monitoring():
    seconds = int(time.monotonic() - started_at)
    with lock:
        average = (round(total_response_ms / successful_predictions, 2)
                   if successful_predictions else None)
        return {
            "status": "running",
            "uptime": f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}",
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "average_response_ms": average,
            "last_prediction": last_prediction
        }
