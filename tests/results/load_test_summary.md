# Load Testing and Simulation Results

## 1. Objective

I carried out load testing to see how my deployed Crop Disease Detection application handles multiple requests. I wanted to check if the application could respond successfully and how long it takes to process requests.

## 2. Test Setup

I used a simple Python script with `requests` and `ThreadPoolExecutor` to send requests to my deployed application on Render.

- **Application URL:** https://crop-disease-detection-b6xr.onrender.com
- **Endpoints tested:** `/health` and `/predict`
- **Test image:** Apple Cedar Apple Rust leaf image
- **Testing methods:** Sequential and concurrent requests

I tested the health endpoint to check if the application was running and the prediction endpoint to check how it handled image predictions.

## 3. Test Results

I performed three tests with different numbers of requests and workers.

| Test | Total Requests | Workers | Successful | Failed |
|---|---:|---:|---:|---:|
| Sequential test | 10 | 1 | 10 | 0 |
| Concurrent test | 10 | 5 | 10 | 0 |
| Final load test | 20 | 5 | 20 | 0 |
| **Total** | **40** | | **40** | **0** |

### Average Response Times

| Test | Health Endpoint | Prediction Endpoint |
|---|---:|---:|
| Sequential test | 0.756 seconds | 1.266 seconds |
| Concurrent test | 0.871 seconds | 2.095 seconds |
| Final load test | 1.233 seconds | 2.518 seconds |

## 4. What I Observed

From the results, I noticed that all 40 requests were successful, with no failed requests.

The prediction endpoint took more time than the health endpoint because it had to process the uploaded image and make a prediction using the CNN model.

I also noticed that the response time increased when I sent multiple requests at the same time. For example, the average prediction response time increased from 1.266 seconds in the sequential test to 2.518 seconds in the final load test.

Even though the response time increased, the application still handled all the requests successfully.

## 5. Conclusion

From these tests, I found that my deployed application was able to handle the small number of sequential and concurrent requests I tested without any failures. I also learned that sending multiple requests at the same time can increase response time.

Overall, the application performed well during my tests.