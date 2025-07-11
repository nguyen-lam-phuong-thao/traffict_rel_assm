import traci
import numpy as np
import random
import timeit
import os
import datetime
# import config

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1 
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7

MAX_WAITING_TIME_RED = 60  # threshold to penalize long red waits

class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, min_green_duration=8, weight_emergency=2, early_stopping_patience=10, early_stopping_delta=1.0):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states + 1  # plus one for current phase duration
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._current_phase_duration = 0
        self._min_green_duration = min_green_duration  # Minimum green phase duration (seconds)
        self._weight_emergency = weight_emergency  # Penalty for emergency braking
        self._previous_action = None
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_delta = early_stopping_delta
        self._early_stopping_counter = 0
        self._best_reward = float('-inf')

    def run(self, episode, epsilon):
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        max_green_duration = 60
        current_action = None
        self._current_phase_duration = 0
        self._previous_action = None

        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            for veh_id in traci.vehicle.getIDList():
                try:
                    acc = traci.vehicle.getAcceleration(veh_id)
                    if acc < -4.0:
                        reward -= self._weight_emergency
                except:
                    pass

            if self._previous_action is not None and current_action is not None and current_action != self._previous_action:
                reward -= 1

            for wait_time in self._waiting_times.values():
                if wait_time > MAX_WAITING_TIME_RED:
                    reward -= 2

            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            if self._current_phase_duration < self._min_green_duration and self._previous_action is not None:
                action = self._previous_action

            if current_action is not None and action == current_action:
                self._current_phase_duration += self._green_duration
                if self._current_phase_duration >= max_green_duration:
                    action = (action + 1) % self._num_actions
                    reward -= 10
                    self._current_phase_duration = 0
            else:
                self._current_phase_duration = 0

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            current_action = action
            self._previous_action = action

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        self._current_phase_duration = 0
        self._previous_action = None
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        # Early stopping check
        if self._sum_neg_reward > self._best_reward + self._early_stopping_delta:
            self._best_reward = self._sum_neg_reward
            self._early_stopping_counter = 0
        else:
            self._early_stopping_counter += 1

        if self._early_stopping_counter >= self._early_stopping_patience:
            print("Early stopping triggered: no improvement for", self._early_stopping_patience, "episodes")
            return simulation_time, training_time, True

        return simulation_time, training_time, False

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        return sum(self._waiting_times.values())

    def _choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        return sum([
            traci.edge.getLastStepHaltingNumber("N2TL"),
            traci.edge.getLastStepHaltingNumber("S2TL"),
            traci.edge.getLastStepHaltingNumber("E2TL"),
            traci.edge.getLastStepHaltingNumber("W2TL")
        ])

    def _get_state(self):
        state = np.zeros(12)
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            route = traci.vehicle.getRoute(car_id)
            if len(route) < 2:
                continue
            from_edge = route[0]
            to_edge = route[1]
            if from_edge == "N2TL":
                if to_edge == "TL2S": state[0] += 1
                elif to_edge == "TL2E": state[1] += 1
                elif to_edge == "TL2W": state[2] += 1
            elif from_edge == "S2TL":
                if to_edge == "TL2N": state[3] += 1
                elif to_edge == "TL2W": state[4] += 1
                elif to_edge == "TL2E": state[5] += 1
            elif from_edge == "E2TL":
                if to_edge == "TL2W": state[6] += 1
                elif to_edge == "TL2N": state[7] += 1
                elif to_edge == "TL2S": state[8] += 1
            elif from_edge == "W2TL":
                if to_edge == "TL2E": state[9] += 1
                elif to_edge == "TL2S": state[10] += 1
                elif to_edge == "TL2N": state[11] += 1
        normalized_duration = self._current_phase_duration / 60
        return np.append(state, normalized_duration)

    def _replay(self):
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_batch(next_states)

            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b
                current_q = q_s_a[i]
                current_q[action] = reward + self._gamma * np.max(q_s_a_d[i])
                x[i] = state
                y[i] = current_q

            self._Model.train_batch(x, y)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
