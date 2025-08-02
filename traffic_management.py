import cv2
import torch
import numpy as np
import random
import gym
import torch.nn as nn
import torch.optim as optim
from collections import deque
from ultralytics import YOLO
import requests
import os
import time
import matplotlib.pyplot as plt

# ----------------- VIDEO DOWNLOAD FUNCTION -----------------
def download_video(url, save_path="traffic_video.mp4"):
    """Downloads video from URL if not already downloaded."""
    if os.path.exists(save_path):
        print(f"✅ Video already exists: {save_path}")
        return save_path

    print("📥 Downloading video...")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"✅ Video downloaded: {save_path}")
    return save_path

# ----------------- ENVIRONMENT SETUP -----------------
class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Actions: [10s, 20s, 30s signal]
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.state = np.array([random.randint(10, 50)], dtype=np.float32)  # Initial traffic density
        self.traffic_history = deque(maxlen=10)

    def get_traffic_density(self, frame):
        """Detect vehicles using YOLO and return traffic density."""
        results = model(frame)
        vehicle_count = sum(1 for obj in results[0].boxes.cls if int(obj) in [2, 3, 5, 7])  # Cars, trucks, buses, motorcycles
        return max(0, vehicle_count)

    def step(self, action, frame):
        """Apply action and return new state, reward, done flag."""
        signal_time = [10, 20, 30][action]
        print(f"🚦 Signal set to {signal_time}s")

        vehicle_count = self.get_traffic_density(frame)
        self.traffic_history.append(vehicle_count)

        avg_traffic = np.mean(self.traffic_history)
        self.state = np.array([avg_traffic], dtype=np.float32)

        # Reward: Encourage reducing congestion
        reward = -avg_traffic + max(0, 15 - abs(signal_time - 20)) * 0.5
        done = False  # Environment does not terminate

        return self.state, reward, done, {}

    def reset(self):
        """Reset environment"""
        self.state = np.array([random.randint(10, 50)], dtype=np.float32)
        self.traffic_history.clear()
        return self.state

# ----------------- DEEP Q-NETWORK (DQN) -----------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 24)  # FIXED size mismatch (previously 32)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)  # 3 actions (10s, 20s, 30s)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------- TRAINING FUNCTION (LIMITED TO 20 SECONDS) -----------------
def train_agent(video_path):
    env = TrafficEnv()
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())  # Sync target model

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    memory = deque(maxlen=5000)
    epsilon = 1.0  # Exploration rate
    gamma = 0.9    # Discount factor
    batch_size = 32

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_limit = 20 * fps  # Process only the first 20 seconds

    traffic_density_over_time = []  # Store traffic density for plotting
    time_stamps = []

    start_time = time.time()
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    for frame_count in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()  # Exploitation

        next_state, reward, done, _ = env.step(action, frame)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Store experience in memory
        memory.append((state, action, reward, next_state, done))

        # Store traffic density for graph
        traffic_density_over_time.append(env.get_traffic_density(frame))
        time_stamps.append(time.time() - start_time)

        # Experience replay training
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q-learning target
            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            q_values = model(states).gather(1, actions).squeeze()

            # Compute loss and optimize
            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

        # Stop after 20 seconds
        if time.time() - start_time >= 20:
            break

    torch.save(model.state_dict(), "traffic_dqn.pth")
    print("✅ Training complete. Model saved.")

    cap.release()
    cv2.destroyAllWindows()

    # --------------- PLOT TRAFFIC DENSITY GRAPH ---------------
    plt.figure(figsize=(10, 5))
    plt.plot(time_stamps, traffic_density_over_time, marker="o", linestyle="-", color="b", label="Traffic Density")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Traffic Density (vehicles detected)")
    plt.title("Traffic Density Over Time (20 sec)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    model = YOLO("yolov5su.pt")  # Load YOLOv5 for vehicle detection

    # 🔹 Provide either a local or online video
    video_url = "https://cdn.pixabay.com/video/2021/11/17/98176-647151506_large.mp4"
    video_path = download_video(video_url)  # Download and use local path

    train_agent(video_path)  # Run training with the downloaded video
