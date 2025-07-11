import traci
import numpy as np
import timeit
import os

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions,
                 min_green_duration=8, weight_emergency=2, max_phase_switch=20, phase_change_warmup=50):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._step = 0
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

        # new logic
        self._min_green_duration = min_green_duration
        self._weight_emergency = weight_emergency
        self._max_phase_switch = max_phase_switch
        self._phase_change_warmup = phase_change_warmup
        self._current_phase_duration = 0
        self._previous_action = None
        self._phase_switch_count = 0

    def run(self, episode_seed):
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode_seed)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._reward_episode = []
        self._queue_length_episode = []
        old_total_wait = 0
        old_action = -1
        current_action = None

        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # phạt phanh gấp
            for veh_id in traci.vehicle.getIDList():
                try:
                    acc = traci.vehicle.getAcceleration(veh_id)
                    if acc < -4.0:
                        reward -= self._weight_emergency
                except:
                    pass

            # chọn action từ mô hình
            action = self._choose_action(current_state)

            # không được đổi pha nếu chưa đạt min green duration
            if self._current_phase_duration < self._min_green_duration and self._previous_action is not None:
                action = self._previous_action

            # đếm số lần đổi pha sau thời gian khởi động
            if self._step > self._phase_change_warmup and current_action is not None and action != current_action:
                self._phase_switch_count += 1
                if self._phase_switch_count > self._max_phase_switch:
                    print("Exceeded max phase switches, stopping early.")
                    break

            # nếu đổi pha -> phase yellow
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_action = action
            old_total_wait = current_total_wait
            current_action = action
            self._previous_action = action
            self._current_phase_duration = (
                self._current_phase_duration + self._green_duration
                if self._previous_action == current_action
                else 0
            )

            self._reward_episode.append(reward)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def _simulate(self, steps_todo):
        if self._step + steps_todo >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            self._queue_length_episode.append(self._get_queue_length())

    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        self._waiting_times = {}
        for car_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
        return sum(self._waiting_times.values())

    def _choose_action(self, state):
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        green_phases = [PHASE_NS_GREEN, PHASE_NSL_GREEN, PHASE_EW_GREEN, PHASE_EWL_GREEN]
        traci.trafficlight.setPhase("TL", green_phases[action_number])

    def _get_queue_length(self):
        return sum([
            traci.edge.getLastStepHaltingNumber("N2TL"),
            traci.edge.getLastStepHaltingNumber("S2TL"),
            traci.edge.getLastStepHaltingNumber("E2TL"),
            traci.edge.getLastStepHaltingNumber("W2TL")
        ])

    def _get_state(self):
        return np.zeros(self._num_states)  # bạn có thể chèn logic như train nếu cần

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
