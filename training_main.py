from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    # Uncomment to check GPU availability
    # print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    # print(tf.config.list_physical_devices('GPU'))

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'] + 1,  # +1 for phase duration
        output_dim=config['num_actions']
    )

    Memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path,
        dpi=96
    )

    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        min_green_duration=config.get('min_green_duration', 8),
        weight_emergency=config.get('weight_emergency', 2),
        early_stopping_patience=config.get('early_stopping_patience', 20),
        early_stopping_delta=config.get('early_stopping_delta', 0.01)
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # Epsilon-greedy strategy

        simulation_time, training_time, early_stopped = Simulation.run(episode, epsilon)

        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')

        if early_stopped:
            print(f"\n>>> Early stopping triggered at episode {episode + 1}")
            break

        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward',xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay',xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue',xlabel='Episode', ylabel='Average queue length (vehicles)')
