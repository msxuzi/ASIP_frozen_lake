import gymnasium as gym  # Importowanie biblioteki gymnasium
import numpy as np       # Importowanie biblioteki numpy
import os                # Importowanie biblioteki os
import matplotlib.pyplot as plt  # do tworzenia wykresów

# Parametry Q-learningu
alpha = 0.1              # współczynnik uczenia (learning rate)
gamma = 0.99             # współczynnik dyskontowania (discount factor)
epsilon = 0.3        #  wartość epsilon

num_episodes = 10000     # liczba epizodów treningowych
max_steps = 200          # maksymalna liczba kroków w epizodzie
q_table_file = "q_table_0.csv"  # nazwa pliku do zapisu Q-tabeli

# Tworzenie środowiska do treningu
train_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Pobranie rozmiaru przestrzeni stanów i akcji
n_states = train_env.observation_space.n  # liczba stanów (dla 4x4: 16)
n_actions = train_env.action_space.n      # liczba akcji (4)

# Próba wczytania istniejącej tablicy Q
if os.path.exists(q_table_file):
    q_table = np.loadtxt(q_table_file, delimiter=",")
    print("Wczytano istniejącą Q-tabelę z pliku.")
else:
    q_table = np.zeros((n_states, n_actions))
    print("Inicjalizacja nowej Q-tabeli.")

# Tablice do monitorowania postępów
cumulative_rewards = []        # lista do przechowywania skumulowanych nagród
steps_all_episodes = []        # liczba kroków w każdym epizodzie
success_all_episodes = []      # 1 jeśli epizod zakończony sukcesem, inaczej 0
q_values_history = []          # zapis Q-tabeli po każdym epizodzie (spłaszczony wektor)
epsilon_history = []           # historia wartości epsilon

# Trenowanie przez określoną liczbę epizodów
for episode in range(num_episodes):
    state, info = train_env.reset()
    done = False
    total_episode_reward = 0
    step = 0
    success = 0

    # Dynamiczne zmniejszanie epsilon
    epsilon_history.append(epsilon)  # zapis wartości epsilon

    for step in range(max_steps):
        # Polityka epsilon-greedy
        if np.random.rand() < epsilon:  # eksploracja
            action = train_env.action_space.sample()
        else:  # eksploatacja
            action = np.argmax(q_table[state, :])

        next_state, reward, done, truncated, info = train_env.step(action)
        best_next_action = np.argmax(q_table[next_state, :])

        # Aktualizacja Q-tabeli
        q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

        total_episode_reward += reward
        state = next_state

        if done:
            if reward == 1.0:
                success = 1
            break

    cumulative_rewards.append(total_episode_reward)  # dodanie całkowitej nagrody do listy
    steps_all_episodes.append(step + 1)
    success_all_episodes.append(success)
    q_values_history.append(q_table.flatten().copy())

# Zapisywanie Q-tabeli
np.savetxt(q_table_file, q_table, delimiter=",")
print("Zapisano Q-tabelę do pliku po treningu.")

# Obliczanie metryk do wykresów
episodes = np.arange(1, len(success_all_episodes) + 1)
window_size = 100

# Średnia krocząca sukcesów
moving_avg_success = np.convolve(success_all_episodes, np.ones(window_size)/window_size, mode='valid')

# Tworzenie wykresów
plt.figure(figsize=(12, 8))

# Rysowanie wykresu skumulowanych nagród od liczby epizodów
plt.subplot(2, 2, 1)
plt.plot(episodes, np.cumsum(cumulative_rewards))
plt.xlabel('Liczba epizodów')
plt.ylabel('Skumulowana nagroda')
plt.title('Skumulowana nagroda od liczby epizodów')

# 2. Odsetek sukcesów
plt.subplot(2, 2, 2)
plt.plot(episodes[:len(moving_avg_success)], moving_avg_success)
plt.title("Odsetek sukcesów (okno=100)")
plt.xlabel("Epizod")
plt.ylabel("Procent sukcesów")

# 3. Liczba kroków na epizod
plt.subplot(2, 2, 3)
plt.plot(episodes, steps_all_episodes)
plt.title("Liczba kroków na epizod")
plt.xlabel("Epizod")
plt.ylabel("Liczba kroków")

# 4. Zmiana wartości epsilon
plt.subplot(2, 2, 4)
plt.plot(episodes, epsilon_history)
plt.title("Zmiana epsilon")
plt.xlabel("Epizod")
plt.ylabel("Wartość epsilon")

plt.tight_layout()
plt.show()


# Po treningu test z wizualizacją
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

for i in range(5):
    state, info = env.reset()
    done = False
    total_reward = 0

    print(f"Uruchamiam środowisko z wyuczonym modelem: Run {i+1}")
    for step in range(max_steps):
        # Wybór najlepszej akcji
        action = np.argmax(q_table[state, :])  
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Epizod zakończony. Wynik: {total_reward}")
            break

env.close()