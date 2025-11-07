import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict
import math
import numpy as np
import torch

# load log data

def read_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        f.close()
    return obj

def load_log(dir, mode="training"):
    if mode == "training":
        file_path = dir + "training_log.pkl"

    elif mode == "playing":
        file_path = dir + "state_log.pkl"

    return read_pkl(file_path)


# plot graph
# def base_plot(dict_logs:dict, list_ex_names:list, legends:dict, num_seed:int, active_plot:list, show_legend:bool):
#     cmap = plt.cm.get_cmap("Paired")
#     colors = cmap(np.linspace(0, 1, len(list_ex_names)))
#     _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
#     x = np.array( dict_logs[list_ex_names[-1]][1]["iteration"] )

#     for i, ex_name in enumerate(list_ex_names):
#         if not(ex_name in active_plot):
#             continue

#         y_reward = np.zeros((x.shape[0], num_seed))
#         y_val_loss = np.zeros((x.shape[0], num_seed))
#         y_sur_loss = np.zeros((x.shape[0], num_seed))
#         y_length = np.zeros((x.shape[0], num_seed))
#         y_success = np.zeros((x.shape[0], num_seed))
#         y_curr = np.zeros((x.shape[0], num_seed))

#         for seed in range(1, num_seed+1):
#             y_reward[:, seed - 1] = np.array( dict_logs[ex_name][seed]["mean_reward"] )
#             y_val_loss[:, seed - 1] = np.array( dict_logs[ex_name][seed]["mean_value_loss"] )
#             y_sur_loss[:, seed - 1] = np.array( dict_logs[ex_name][seed]["mean_surrogate_loss"] )
#             y_length[:, seed - 1] = np.array( dict_logs[ex_name][seed]["mean_episode_length"] )
#             y_success[:, seed -1] = np.array(torch.tensor( dict_logs[ex_name][seed]["reach_goal_percent"] ).tolist())
#             if "radius_level" in list(dict_logs[ex_name][seed].keys()):
#                 y_curr[:, seed - 1] = np.array( dict_logs[ex_name][seed]["radius_level"] )

#         y_reward_mean = np.mean(y_reward, axis=1)
#         y_val_loss_mean = np.mean(y_val_loss, axis=1)
#         y_sur_loss_mean = np.mean(y_sur_loss, axis=1)
#         y_length_mean = np.mean(y_length, axis=1)
#         y_success_mean = np.mean(y_success, axis=1)

#         y_reward_std = np.std(y_reward, axis=1)
#         y_val_loss_std = np.std(y_val_loss, axis=1)
#         y_sur_loss_std = np.std(y_sur_loss, axis=1)
#         y_length_std = np.std(y_length, axis=1)
#         y_success_std = np.std(y_success, axis=1)

#         confidence_level = 0.95
#         z_score = np.quantile(np.random.standard_normal(size=1000), 1 - (1 - confidence_level) / 2)

#         y_reward_upper_bound = y_reward_mean + z_score * y_reward_std
#         y_reward_lower_bound = y_reward_mean - z_score * y_reward_std

#         y_val_loss_upper_bound = y_val_loss_mean + z_score * y_val_loss_std
#         y_val_loss_lower_bound = y_val_loss_mean - z_score * y_val_loss_std

#         y_sur_loss_upper_bound = y_sur_loss_mean + z_score * y_sur_loss_std
#         y_sur_loss_lower_bound = y_sur_loss_mean - z_score * y_sur_loss_std

#         y_length_upper_bound = y_length_mean + z_score * y_length_std
#         y_length_lower_bound = y_length_mean - z_score * y_length_std

#         y_success_upper_bound = y_success_mean + z_score * y_success_std
#         y_success_lower_bound = y_success_mean - z_score * y_success_std

#         # mean reward
#         ax[0][0].plot(x, y_reward_mean, label=legends[ex_name], color=colors[i])
#         ax[0][0].fill_between(x, y_reward_upper_bound, y_reward_lower_bound, alpha=0.2, color=colors[i])
#         ax[0][0].set_title("Mean reward")
#         ax[0][0].set(xlabel='iteration', ylabel='reward')
#         if show_legend: ax[0][0].legend()

#         # radius level
#         if "radius_level" in list(dict_logs[ex_name][1].keys()):
#             y_curr_mean = np.mean(y_curr, axis=1)
#             y_curr_std = np.std(y_curr, axis=1)
#             y_curr_upper_bound = y_curr_mean + z_score * y_curr_std
#             y_curr_lower_bound = y_curr_mean - z_score * y_curr_std

#             ax[0][1].plot(x, y_curr_mean, label=legends[ex_name], color=colors[i])
#             ax[0][1].fill_between(x, y_curr_upper_bound, y_curr_lower_bound, alpha=0.2, color=colors[i])
#             ax[0][1].set_title("Curriculum radius")
#             ax[0][1].set(xlabel='iteration', ylabel='radius (m)')
#             if show_legend: ax[0][1].legend()
#         else:
#             ax[0][1].plot(x, [6 for _ in x], label=legends[ex_name], color=colors[i])
#             ax[0][1].set_title("Curriculum radius")
#             ax[0][1].set(xlabel='iteration', ylabel='radius (m)')
#             if show_legend: ax[0][1].legend()
        
#         # success rate
#         ax[0][2].plot(x, y_success_mean, label=legends[ex_name], color=colors[i])
#         ax[0][2].fill_between(x, y_success_upper_bound, y_success_lower_bound, alpha=0.2, color=colors[i])
#         ax[0][2].set_title("Success rate")
#         ax[0][2].set(xlabel='iteration', ylabel='success rate')
#         if show_legend: ax[0][2].legend()
        
#         # value loss
#         ax[1][0].plot(x, y_val_loss_mean, label=legends[ex_name], color=colors[i])
#         ax[1][0].fill_between(x, y_val_loss_upper_bound, y_val_loss_lower_bound, alpha=0.2, color=colors[i])
#         ax[1][0].set_title("Value Loss")
#         ax[1][0].set(xlabel='iteration', ylabel='loss')
#         if show_legend: ax[1][0].legend()

#         # surrogate loss
#         ax[1][1].plot(x, y_sur_loss_mean, label=legends[ex_name], color=colors[i])
#         ax[1][1].fill_between(x, y_sur_loss_upper_bound, y_sur_loss_lower_bound, alpha=0.2, color=colors[i])
#         ax[1][1].set_title("Surrogate Loss")
#         ax[1][1].set(xlabel='iteration', ylabel='loss')
#         if show_legend: ax[1][1].legend()
        
#         # surrogate loss
#         ax[1][2].plot(x, y_length_mean, label=legends[ex_name], color=colors[i])
#         ax[1][2].fill_between(x, y_length_upper_bound, y_length_lower_bound, alpha=0.2, color=colors[i])
#         ax[1][2].set_title("Episode length")
#         ax[1][2].set(xlabel='iteration', ylabel='episode length (step)')
#         if show_legend: ax[1][2].legend()
        
#     plt.show()


def base_plot(dict_logs: dict, list_ex_names: list, legends: dict, num_seed: int, active_plot: list, show_legend: bool):
    cmap = plt.cm.get_cmap("Paired")
    colors = cmap(np.linspace(0, 1, len(list_ex_names)))
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    x = np.array(dict_logs[list_ex_names[-1]][1]["iteration"])

    all_lines = []
    all_labels = []

    for i, ex_name in enumerate(list_ex_names):
        if ex_name not in active_plot:
            continue

        y_reward = np.zeros((x.shape[0], num_seed))
        y_val_loss = np.zeros((x.shape[0], num_seed))
        y_sur_loss = np.zeros((x.shape[0], num_seed))
        y_length = np.zeros((x.shape[0], num_seed))
        y_success = np.zeros((x.shape[0], num_seed))
        y_curr = np.zeros((x.shape[0], num_seed))

        for seed in range(1, num_seed + 1):
            y_reward[:, seed - 1] = np.array(dict_logs[ex_name][seed]["mean_reward"])
            y_val_loss[:, seed - 1] = np.array(dict_logs[ex_name][seed]["mean_value_loss"])
            y_sur_loss[:, seed - 1] = np.array(dict_logs[ex_name][seed]["mean_surrogate_loss"])
            y_length[:, seed - 1] = np.array(dict_logs[ex_name][seed]["mean_episode_length"])
            y_success[:, seed - 1] = np.array(torch.tensor(dict_logs[ex_name][seed]["reach_goal_percent"]).tolist())
            if "radius_level" in dict_logs[ex_name][seed]:
                y_curr[:, seed - 1] = ( np.array(dict_logs[ex_name][seed]["radius_level"]) - 1 ) * 4 # there are 20 level in range [1, 6] with step 0.25

        # Mean and Std
        y_reward_mean = np.mean(y_reward, axis=1)
        y_val_loss_mean = np.mean(y_val_loss, axis=1)
        y_sur_loss_mean = np.mean(y_sur_loss, axis=1)
        y_length_mean = np.mean(y_length, axis=1)
        y_success_mean = np.mean(y_success, axis=1)

        y_reward_std = np.std(y_reward, axis=1)
        y_val_loss_std = np.std(y_val_loss, axis=1)
        y_sur_loss_std = np.std(y_sur_loss, axis=1)
        y_length_std = np.std(y_length, axis=1)
        y_success_std = np.std(y_success, axis=1)

        # Confidence bounds
        confidence_level = 0.95
        z_score = np.quantile(np.random.standard_normal(size=1000), 1 - (1 - confidence_level) / 2)

        def bounds(mean, std): return mean + z_score * std, mean - z_score * std

        reward_ub, reward_lb = bounds(y_reward_mean, y_reward_std)
        val_loss_ub, val_loss_lb = bounds(y_val_loss_mean, y_val_loss_std)
        sur_loss_ub, sur_loss_lb = bounds(y_sur_loss_mean, y_sur_loss_std)
        length_ub, length_lb = bounds(y_length_mean, y_length_std)
        success_ub, success_lb = bounds(y_success_mean, y_success_std)

        # reward
        line, = ax[0][0].plot(x, y_reward_mean, label=legends[ex_name], color=colors[i])
        ax[0][0].fill_between(x, reward_ub, reward_lb, alpha=0.2, color=colors[i])
        ax[0][0].set_title("Mean reward")
        ax[0][0].set(xlabel='iteration', ylabel='reward')
        all_lines.append(line)
        all_labels.append(legends[ex_name])

        # curriculum radius
        if "radius_level" in dict_logs[ex_name][1]:
            y_curr_mean = np.mean(y_curr, axis=1)
            y_curr_std = np.std(y_curr, axis=1)
            curr_ub, curr_lb = bounds(y_curr_mean, y_curr_std)
            ax[0][1].plot(x, y_curr_mean, color=colors[i])
            ax[0][1].fill_between(x, curr_ub, curr_lb, alpha=0.2, color=colors[i])
        else:
            ax[0][1].plot(x, [20 for _ in x], color=colors[i])
        ax[0][1].set_title("Curriculum Level")
        ax[0][1].set(xlabel='iteration', ylabel='level')

        # success rate
        ax[0][2].plot(x, y_success_mean, color=colors[i])
        ax[0][2].fill_between(x, success_ub, success_lb, alpha=0.2, color=colors[i])
        ax[0][2].set_title("Success Rate")
        ax[0][2].set(xlabel='iteration', ylabel='success rate')

        # value loss
        ax[1][0].plot(x, y_val_loss_mean, color=colors[i])
        ax[1][0].fill_between(x, val_loss_ub, val_loss_lb, alpha=0.2, color=colors[i])
        ax[1][0].set_title("Value Loss")
        ax[1][0].set(xlabel='iteration', ylabel='loss')

        # surrogate loss
        ax[1][1].plot(x, y_sur_loss_mean, color=colors[i])
        ax[1][1].fill_between(x, sur_loss_ub, sur_loss_lb, alpha=0.2, color=colors[i])
        ax[1][1].set_title("Surrogate Loss")
        ax[1][1].set(xlabel='iteration', ylabel='loss')

        # episode length
        ax[1][2].plot(x, y_length_mean, color=colors[i])
        ax[1][2].fill_between(x, length_ub, length_lb, alpha=0.2, color=colors[i])
        ax[1][2].set_title("Episode Length")
        ax[1][2].set(xlabel='iteration', ylabel='episode length (step)')

    if show_legend:
        fig.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(0.97, 0.88), fontsize='large')
        plt.subplots_adjust(right=0.97)  # Make room at the top

    plt.show()

# def plot_navigation(dict_state_log:dict, run_name:str, run_label:str):
#     nb_rows = 3
#     nb_cols = 1
#     _, axs = plt.subplots(nb_rows, nb_cols, figsize=(5, 15))

#     dt = dict_state_log[run_name]['dt'][0]
#     num_samples = dict_state_log[run_name]['num_samples'][0]
#     num_step = len(dict_state_log[run_name]['base_vel_x'])

#     time = np.linspace(0, dt*num_samples, num_step)
#     a = axs[0]
#     if dict_state_log[run_name]["command_x"]: a.plot(time, dict_state_log[run_name]["command_x"], label='Command')
#     if dict_state_log[run_name]["robot_x"]: a.plot(time, dict_state_log[run_name]["robot_x"], label=run_label)
#     a.set(xlabel='time [s]', ylabel='x position [m]')
#     a.set_title('Position (x)', fontweight ="bold")
#     a.set_ylim(-9.5, 9.5)
#     a.legend()

#     a = axs[1]
#     if dict_state_log[run_name]["command_y"]: a.plot(time, dict_state_log[run_name]["command_y"], label='Command')
#     if dict_state_log[run_name]["robot_y"]: a.plot(time, dict_state_log[run_name]["robot_y"], label=run_label)
#     a.set(xlabel='time [s]', ylabel='y position [m]')
#     a.set_title('Position (y)', fontweight ="bold")
#     a.set_ylim(-9.5, 9.5)
#     a.legend()

#     a = axs[2]
#     if dict_state_log[run_name]["command_x"]: a.plot(time, dict_state_log[run_name]["command_z"], label='Command')
#     if dict_state_log[run_name]["robot_z"]: a.plot(time, dict_state_log[run_name]["robot_z"], label=run_label)
#     a.set(xlabel='time [s]', ylabel='z position [m]')
#     a.set_title('Position (z)', fontweight ="bold")
#     a.set_ylim(0.2, 1)
#     a.legend()

#     plt.show()
    
def plot_navigation(dict_state_log:dict, run_name:str, run_label:str):
    nb_rows = 1
    nb_cols = 3
    _, axs = plt.subplots(nb_rows, nb_cols, figsize=(15, 5))

    dt = dict_state_log[run_name]['dt'][0]
    num_samples = dict_state_log[run_name]['num_samples'][0]
    num_step = len(dict_state_log[run_name]['base_vel_x'])

    time = np.linspace(0, dt*num_samples, num_step)
    a = axs[0]
    if dict_state_log[run_name]["command_x"]: a.plot(time, dict_state_log[run_name]["command_x"], label='Command')
    if dict_state_log[run_name]["robot_x"]: a.plot(time, dict_state_log[run_name]["robot_x"], label=run_label)
    a.set(xlabel='time [s]', ylabel='x position [m]')
    a.set_title('Position (x)', fontweight ="bold")
    # a.set_ylim(-9.5, 9.5)
    a.legend()

    a = axs[1]
    if dict_state_log[run_name]["command_y"]: a.plot(time, dict_state_log[run_name]["command_y"], label='Command')
    if dict_state_log[run_name]["robot_y"]: a.plot(time, dict_state_log[run_name]["robot_y"], label=run_label)
    a.set(xlabel='time [s]', ylabel='y position [m]')
    a.set_title('Position (y)', fontweight ="bold")
    # a.set_ylim(-9.5, 9.5)
    a.legend()

    a = axs[2]
    if dict_state_log[run_name]["command_x"]: a.plot(time, dict_state_log[run_name]["command_z"], label='Command')
    if dict_state_log[run_name]["robot_z"]: a.plot(time, dict_state_log[run_name]["robot_z"], label=run_label)
    a.set(xlabel='time [s]', ylabel='z position [m]')
    a.set_title('Position (z)', fontweight ="bold")
    a.set_ylim(0.2, 1)
    a.legend()

    plt.show()