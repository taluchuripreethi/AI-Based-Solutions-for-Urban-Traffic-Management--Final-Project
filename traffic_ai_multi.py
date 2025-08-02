import os
import cv2
import time
import torch
import random
import folium
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
from ultralytics import YOLO
from collections import deque
from torch import nn, optim
from folium.plugins import HeatMap

# Configuration
VIDEO_DIR = "videos"
YOLO_MODEL_PATH = "yolov8n.pt"
TRAINING_TIME = 10
GAMMA = 0.9
EPSILON = 1.0
BATCH_SIZE = 32
MEMORY_SIZE = 1000
LEARNING_RATE = 0.001
TRAFFIC_THRESHOLDS = {
    "low": 10,
    "medium": 20
}
AREA_LOCATIONS = {
    "area1": (17.385044, 78.486671),
    "area3": (17.38, 78.49),
    "BB_dfd22c45": (17.388, 78.487)
}

# DQN Model
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        )

    def forward(self, x):
        return self.fc(x)

# Traffic Environment
class TrafficEnv:
    def __init__(self, model):
        self.model = model
        self.state = np.array([random.randint(5, 30)], dtype=np.float32)
        self.history = deque(maxlen=10)

    def get_density(self, frame):
        results = self.model(frame, verbose=False)
        return sum(1 for c in results[0].boxes.cls if int(c) in [2, 3, 5, 7])

    def step(self, action, frame):
        wait_time = [10, 20, 30][action]
        density = self.get_density(frame)
        self.history.append(density)
        avg_density = np.mean(self.history)
        reward = -avg_density + max(0, 15 - abs(wait_time - 20)) * 0.5
        self.state = np.array([avg_density], dtype=np.float32)
        return self.state, reward, False, density

    def reset(self):
        self.history.clear()
        self.state = np.array([random.randint(5, 30)], dtype=np.float32)
        return self.state

# Train DQN per area
def train_area(name, video_path, model, dqn_model):
    env = TrafficEnv(model)
    optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open {video_path}")
        return dqn_model, 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = fps * TRAINING_TIME

    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    densities, timestamps = [], []
    start_time = time.time()

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        action = random.randint(0, 2) if random.random() < EPSILON else torch.argmax(dqn_model(state)).item()
        next_state, reward, done, density = env.step(action, frame)
        densities.append(density)
        timestamps.append(time.time() - start_time)

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        memory.append((state, action, reward, next_state, done))

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            s, a, r, ns, d = zip(*batch)
            s = torch.cat(s)
            a = torch.tensor(a).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32)
            ns = torch.cat(ns)
            d = torch.tensor(d, dtype=torch.bool)

            q_next = dqn_model(ns).max(1)[0]
            targets = r + GAMMA * q_next * (~d).float()
            q_values = dqn_model(s).gather(1, a).squeeze()
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    cap.release()
    save_density_plot(name, timestamps, densities)
    return dqn_model, max(densities)

# Save density plots
def save_density_plot(area, times, densities):
    plt.figure()
    plt.plot(times, densities, label=area)
    plt.title(f"Traffic Density - {area}")
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.savefig(f"traffic_density_plot_{area}.png")
    plt.close()

# Draw colored map
def draw_traffic_map(density_by_area):
    center = [17.385044, 78.486671]
    m = folium.Map(location=center, zoom_start=14)

    for area, density in density_by_area.items():
        lat, lon = AREA_LOCATIONS.get(area, center)
        if density >= TRAFFIC_THRESHOLDS["medium"]:
            color = 'red'
        elif density >= TRAFFIC_THRESHOLDS["low"]:
            color = 'orange'
        else:
            color = 'green'

        folium.Marker(
            [lat, lon],
            tooltip=f"{area}: {density} vehicles",
            icon=folium.Icon(color=color)
        ).add_to(m)

    m.save("traffic_map.html")
    print("🗺️  Traffic map saved as traffic_map.html")

# Suggest alternate route
def suggest_alternate_route(area_name):
    print(f"🚨 High traffic detected in {area_name}!")
    print("🔁 Suggesting alternate route...")
    base_url = "https://www.google.com/maps/dir/"
    start = "Current+Location"
    end = f"{area_name}+Hyderabad"
    webbrowser.open(f"{base_url}{start}/{end}")

# Main
if __name__ == "__main__":
    print("🚦 Starting AI Traffic Management...\n")
    model = YOLO(YOLO_MODEL_PATH)
    shared_dqn = DQN()
    density_by_area = {}

    input_videos = {
        "area1": os.path.join(VIDEO_DIR, "area1.mp4.mov"),
        "area3": os.path.join(VIDEO_DIR, "area3.mp4.mov"),
        "BB_dfd22c45": os.path.join(VIDEO_DIR, "area2.mp4.mov")
    }

    area_to_check = input("📍 Enter the area name to check traffic (e.g., area1): ").strip()

    if area_to_check in input_videos:
        print(f"📹 Processing {area_to_check}...")
        shared_dqn, max_density = train_area(area_to_check, input_videos[area_to_check], model, shared_dqn)
        density_by_area[area_to_check] = max_density

        if max_density >= TRAFFIC_THRESHOLDS["medium"]:
            suggest_alternate_route(area_to_check)
        else:
            print(f"✅ Traffic is normal in {area_to_check}. No alternate route needed.")
    else:
        print("❌ Area not found. Please check the area name and try again.")

    draw_traffic_map(density_by_area)
    torch.save(shared_dqn.state_dict(), "traffic_dqn.pth")
    print("✅ AI model saved as traffic_dqn.pth")
