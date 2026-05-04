import requests
import random

URL = "http://127.0.0.1:8888/infer"

# Normal data
for _ in range(35):
    data = {
        "pipe_age_years": random.randint(1, 40),
        "flow_rate_lph": random.uniform(100, 5000),
        "moisture_pct": random.uniform(10, 90),
        "wall_thickness_mm": random.uniform(2, 15)
    }
    requests.post(URL, json=data)

# Drifted data
for _ in range(15):
    data = {
        "pipe_age_years": random.randint(40, 80),
        "flow_rate_lph": random.uniform(100, 5000),
        "moisture_pct": random.uniform(100, 180),
        "wall_thickness_mm": random.uniform(2, 15)
    }
    requests.post(URL, json=data)

print("Traffic sent")