import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import cv2
import folium
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
from folium import plugins

# ---------------- CONFIG ----------------
VIDEO_FOLDER = "videos"
YOLO_MODEL_PATH = "yolov8n.pt"
OUTPUT_MAP = "traffic_map.html"
TRAINING_TIME = 10
GAMMA = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEMORY_SIZE = 5000
TRAFFIC_THRESHOLDS = {"low": 10, "medium": 25}  # traffic level cutoffs

AREA_LOCATIONS = {
    "area1": [17.385044, 78.486671],
    "area2": [17.392044, 78.481671],
    "area3": [17.379044, 78.490671],
}

# ---------------- ENVIRONMENT ----------------
class TrafficEnv(gym.Env):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.state = np.array([random.randint(10, 50)], dtype=np.float32)
        self.traffic_history = deque(maxlen=10)

    def get_traffic_density(self, frame):
        results = self.model(frame, verbose=False)
        vehicle_count = sum(1 for c in results[0].boxes.cls if int(c) in [2, 3, 5, 7])
        return vehicle_count

    def step(self, action, frame):
        signal_time = [10, 20, 30][action]
        vehicle_count = self.get_traffic_density(frame)
        self.traffic_history.append(vehicle_count)
        avg_traffic = np.mean(self.traffic_history)
        self.state = np.array([avg_traffic], dtype=np.float32)
        reward = -avg_traffic + max(0, 15 - abs(signal_time - 20)) * 0.5
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([random.randint(10, 50)], dtype=np.float32)
        self.traffic_history.clear()
        return self.state

# ---------------- DQN ----------------
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- PLOTTING ----------------
def plot_traffic_density(area, timestamps, densities):
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, densities, label=f"{area}", color='blue')
    plt.title(f"{area} - Traffic Density Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles Detected")
    plt.grid(True)
    plt.tight_layout()
    filename = f"traffic_density_plot_{area}.png"
    plt.savefig(filename)
    print(f"📊 Saved {filename}")
    plt.close()

# ---------------- TRAIN AREA ----------------
def train_area(area, video_path, yolo_model, shared_dqn=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open {video_path}")
        return None, 0

    env = TrafficEnv(yolo_model)
    dqn = shared_dqn or DQN()
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_limit = TRAINING_TIME * fps

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    traffic_density = []
    time_stamps = []
    start_time = time.time()

    for _ in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            break

        action = random.choice([0, 1, 2])
        next_state, reward, done, _ = env.step(action, frame)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        vehicle_count = env.get_traffic_density(frame)

        memory.append((state, action, reward, next_state, done))
        traffic_density.append(vehicle_count)
        time_stamps.append(time.time() - start_time)

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.bool)

            next_q = dqn(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q * (~dones).float()
            q_values = dqn(states).gather(1, actions).squeeze()
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    cap.release()
    plot_traffic_density(area, time_stamps, traffic_density)
    return dqn, max(traffic_density) if traffic_density else 0

# ---------------- SHOW MAP ----------------
def get_color_for_density(density):
    if density <= TRAFFIC_THRESHOLDS["low"]:
        return "green"
    elif density <= TRAFFIC_THRESHOLDS["medium"]:
        return "orange"
    else:
        return "red"

def show_map(area_data):
    center = AREA_LOCATIONS["area1"]
    fmap = folium.Map(location=center, zoom_start=14)
    for area, (lat, lon, density) in area_data.items():
        color = get_color_for_density(density)
        folium.Marker(
            location=[lat, lon],
            popup=f"{area}: {density} vehicles",
            icon=folium.Icon(color=color)
        ).add_to(fmap)

    fmap.save(OUTPUT_MAP)
    print(f"🗺️  Traffic map saved as {OUTPUT_MAP}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚦 Starting AI Traffic Management...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    area_data = {}

    for area, coords in AREA_LOCATIONS.items():
        video_file = os.path.join(VIDEO_FOLDER, f"{area}.mp4")
        print(f"\n📹 Processing {area}...")
        dqn_model, max_density = train_area(area, video_file, yolo_model)
        area_data[area] = (*coords, max_density)

    show_map(area_data)
