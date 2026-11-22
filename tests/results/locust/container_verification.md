# Docker Container Verification

The Locust test ran with 5 simulated users for 30 seconds.

## Container 1
- Successful predictions: 33
- Failed predictions: 0
- Average internal response time: 335.82 ms

## Container 2
- Successful predictions: 34
- Failed predictions: 0
- Average internal response time: 354.46 ms

## Conclusion
Both Docker model containers received prediction requests through Nginx. The 67 successful predictions match the Locust two-container test results.
