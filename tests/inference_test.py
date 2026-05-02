# time_test.py
import time
from predict import predict

request = {
    "pickup_zone": 132,
    "dropoff_zone": 161,
    "requested_at": "2024-01-15T08:30:00",
    "passenger_count": 1,
}

# Warm up — first call loads model.pkl
predict(request)

# Time 1000 calls
start = time.perf_counter()
for _ in range(1000):
    predict(request)
elapsed = time.perf_counter() - start

avg_ms = (elapsed / 1000) * 1000
print(f"Average inference time: {avg_ms:.2f} ms")
print(f"{'PASS' if avg_ms < 200 else 'FAIL'} — limit is 200ms")