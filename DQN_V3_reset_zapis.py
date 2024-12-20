import os
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt

# Parametry DQN
num_episodes = 10000      # liczba epizodów testowych
max_steps = 200           # maksymalna liczba kroków w epizodzie
window_size = 100         # rozmiar okna dla średniej kroczącej

total_timesteps = 100000  # Znacznie więcej kroków uczenia dla lepszego dopasowania
learning_rate = 1e-4      # Jeszcze mniejszy krok uczenia dla stabilności
batch_size = 128          # Duża partia danych dla efektywnego uczenia
target_update_interval = 10000  # Rzadsze aktualizowanie target network dla stabilizacji
exploration_fraction = 0.3      # Dłuższa eksploracja na początku
exploration_final_eps = 0.005   # Bardzo mały epsilon dla ostatecznej eksploatacji

gamma = 0.90              # Jeszcze mniejsze gamma dla szybszego reagowania na nagrody

# Tworzenie środowiska do treningu
train_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Ścieżka do zapisu modelu
model_path = "dqn_frozenlake_model.zip"

def reset_model(env):
    """
    Funkcja resetująca model poprzez stworzenie nowej instancji i wytrenowanie go ponownie.
    """
    global model
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=batch_size,
        tau=0.05,
        gamma=gamma,
        train_freq=4,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=1
    )
    print("Resetowanie i ponowne trenowanie modelu...")
    model.learn(total_timesteps=total_timesteps)
    print("Trening zakończony po resecie!")
    model.save(model_path)  # Zapis modelu po resecie
    print(f"Model zapisany w {model_path}")

# Sprawdzanie, czy istnieje zapisany model
if os.path.exists(model_path):
    print(f"Znaleziono zapisany model: {model_path}. Czy go załadować? (tak/nie)")
    user_input = input().strip().lower()
    if user_input == 'tak':
        model = DQN.load(model_path, env=train_env)
        print("Model załadowany.")
    else:
        print("Model nie zostanie załadowany. Rozpoczęcie treningu nowego modelu.")
        reset_model(train_env)
else:
    print("Nie znaleziono zapisanego modelu. Rozpoczęcie treningu nowego modelu.")
    reset_model(train_env)

# Monitorowanie statystyk
cumulative_rewards = []
steps_all_episodes = []
success_all_episodes = []

# Testowanie wytrenowanego modelu
for episode in range(num_episodes):
    state, info = train_env.reset()
    done = False
    total_episode_reward = 0
    step = 0
    success = 0

    for step in range(max_steps):
        action, _states = model.predict(state, deterministic=False)
        action = int(action)
        next_state, reward, done, truncated, info = train_env.step(action)

        total_episode_reward += reward
        state = next_state

        if done:
            if reward == 1.0:
                success = 1
            break

    cumulative_rewards.append(total_episode_reward)
    steps_all_episodes.append(step + 1)
    success_all_episodes.append(success)

# Obliczanie metryk do wykresów
episodes = np.arange(1, len(success_all_episodes) + 1)
moving_avg_success = np.convolve(success_all_episodes, np.ones(window_size)/window_size, mode='valid')

# Tworzenie wykresów
plt.figure(figsize=(12, 8))

# Skumulowana nagroda
plt.subplot(2, 2, 1)
plt.plot(episodes, np.cumsum(cumulative_rewards))
plt.xlabel('Liczba epizodów')
plt.ylabel('Skumulowana nagroda')
plt.title('Skumulowana nagroda od liczby epizodów')

# Odsetek sukcesów
plt.subplot(2, 2, 2)
plt.plot(episodes[:len(moving_avg_success)], moving_avg_success)
plt.title("Odsetek sukcesów (okno=100)")
plt.xlabel("Epizod")
plt.ylabel("Procent sukcesów")

# Liczba kroków na epizod
plt.subplot(2, 2, 3)
plt.plot(episodes, steps_all_episodes)
plt.title("Liczba kroków na epizod")
plt.xlabel("Epizod")
plt.ylabel("Liczba kroków")

plt.tight_layout()
plt.show()
