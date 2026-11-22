import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://crop-disease-detection-b6xr.onrender.com"


def send_request(endpoint, image_path=None):
    start = time.perf_counter()

    try:
        if image_path:
            with open(image_path, "rb") as image:
                response = requests.post(
                    BASE_URL + endpoint,
                    files={"image": (image_path.name, image)},
                    timeout=120
                )
        else:
            response = requests.get(BASE_URL + endpoint, timeout=120)

        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response_time": round(time.perf_counter() - start, 3)
        }

    except requests.RequestException as error:
        return {
            "status_code": None,
            "success": False,
            "response_time": round(time.perf_counter() - start, 3),
            "error": str(error)
        }


def run_test(endpoint, total_requests, workers, image_path=None):
    print(f"\nTesting {endpoint}")
    print(f"Requests: {total_requests}")
    print(f"Concurrent workers: {workers}")

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            lambda _: send_request(endpoint, image_path),
            range(total_requests)
        ))

    duration = time.perf_counter() - start
    response_times = [result["response_time"] for result in results]
    successful = sum(result["success"] for result in results)

    summary = {
        "endpoint": endpoint,
        "total_requests": total_requests,
        "successful_requests": successful,
        "failed_requests": total_requests - successful,
        "success_rate_percent": round(successful / total_requests * 100, 2),
        "concurrent_workers": workers,
        "total_duration_seconds": round(duration, 3),
        "average_response_seconds": round(sum(response_times) / len(response_times), 3),
        "minimum_response_seconds": min(response_times),
        "maximum_response_seconds": max(response_times),
        "requests_per_second": round(total_requests / duration, 2),
        "status_codes": [result["status_code"] for result in results]
    }

    print(json.dumps(summary, indent=4))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        print("Image file not found:", image_path)
        return

    if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        print("Please select a JPG or PNG image.")
        return

    if args.requests < 1 or args.workers < 1 or args.workers > args.requests:
        print("Invalid number of requests or workers.")
        return

    health_results = run_test("/health", args.requests, args.workers)
    prediction_results = run_test("/predict", args.requests, args.workers, image_path)

    output = {
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "health_test": health_results,
        "prediction_test": prediction_results
    }

    output_folder = Path("tests/results")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "load_test_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()