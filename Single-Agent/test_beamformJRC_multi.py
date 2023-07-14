# %pdb on

"""

Run an agent that uses the "Heuristic" or "Round Robin" policies. Running instructions in the README file.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González,
"Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication,"
2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.


"""
from __future__ import division



# -*- coding: utf-8 -*-
"""

Dual carriageway JRC game environment.

J. Lee, Y. Cheng, D. Niyato, Y. L. Guan and G. David González,
"Deep Reinforcement Learning for Time Allocation and Directional Transmission in Joint Radar-Communication,"
2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2559-2564, doi: 10.1109/WCNC51071.2022.9771580.


"""

import gym
import numpy as np
from scipy.stats import norm
from gym import spaces
from gym.utils import seeding

# from beamformJRCenv_1lane import Beamform_JRC
# from .beamformJRCenv_vd import Beamform_JRC
from gym.wrappers import Monitor
import numpy as np
import numpy.random as random
import random as python_random
# import json
import time
import os
import argparse
import logz
import inspect


def SINR(P_signal, noise, P_interferences):
    sum_P_interference = np.sum(P_interferences)
    ans = P_signal / (np.power(noise, 2) + sum_P_interference + 1e-7)
    
    return ans

def power_received(P_transmit, R, wavelength=1, gain=1):
    P_received = (P_transmit * (gain**2) * (wavelength**2)) / (np.power(4*np.pi, 2) * np.power(R, 2) + 1e-7)
    
    return P_received

def success_rate(SINR):
    BER = 1- norm.cdf(np.sqrt(2*SINR))     # Bit Error Rate
    return (1 - BER) * (SINR > 0)

def vector_angle(vector1, vector2):
    ''' returns angle between vector1 and vector2 in degrees
    vector 1: vector from ego vehicle
    vector 2: comparison vectors from other N-1 vehicles
    '''
    unit_vector1 = vector1 / (np.linalg.norm(vector1) + 1e-7)
    unit_vector2 = vector2 / (np.linalg.norm(vector2, axis=1) + 1e-7)[:,None]
    cos = np.dot(unit_vector2, unit_vector1)        # commutative: a.b = b.a
    sin = np.cross(unit_vector2, unit_vector1)      # anti-clockwise angle from vector 2 to vector 1
    angle = np.arctan2(sin,cos)
    
    return (angle / np.pi * 180)

def rotation(vector, direction):
    '''
    Input: vector to be rotated, (direction X pi) to be rotated by
    

    Parameters
    ----------
    vector : TYPE
        DESCRIPTION.
    direction : TYPE
        DESCRIPTION.

    Returns
    -------
    rotated_vector : TYPE
        DESCRIPTION.

    '''
    theta = direction * (np.pi/2)
    cos = np.cos(theta)
    sin = np.sin(theta)
    R = np.array([[cos, -sin], [sin, cos]])     # rotation matrix
    rotated_vector = np.matmul(R, vector)
    return rotated_vector

class Beamform_JRC(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    
    def __init__(self, env_config):
        self.seed(123)
        self.viewer = None
        
        self.timestep = 0.5     # 0.5 seconds per time step
        self.N = env_config['num_users']      # num users
        self.N_obs = env_config['num_users_NN']       # num users observed by NN
        self.N_RSU = 8  # num roadside units
        self.v = 15     # 14 m/s
        self.v_diff = 7.5
        self.noise = 0.0007     # standard variation of noise (sigma)
        self.ob_time = env_config['ob_time']
        
        ''' env dimensions '''
        self.max_x_dim = env_config['x_dim']    # length of road
        self.min_x = 0
        self.min_x_gap = 5
        self.lane_width = 4
        self.num_lanes = env_config['num_lanes']  # number of lanes per direction
        self.max_y_dim = (self.lane_width * self.num_lanes) * 2    # width of env
        
        self.x_width = 7.5    # metres per unit for data grid
        self.y_width = 4 # 2
        
        self.w_comm = env_config['w_comm']
        self.w_rad = env_config['w_rad']
        
        self.veh_enter_rate = (self.N * self.v) / self.max_x_dim
        self.new_veh_queue = 0
        
        
        '''
        new data legend:
            1 - front
            2 - left
            3 - back
            4 - right

        '''
        self.received_mask = np.array([[2,2,2,2,2,2,2],
                                       [2,2,2,2,2,2,2],
                                       [3,3,3,1,1,1,1],
                                       [4,4,4,4,4,4,4],
                                       [4,4,4,4,4,4,4]])
        self.new_data_mask = np.array([[0,0,0,2,0,0,0],
                                       [0,0,0,2,0,0,0],
                                       [0,0,3,0,1,1,0],
                                       [0,0,0,4,0,0,0],
                                       [0,0,0,4,0,0,0]])
        self.new_data_mask_flipped = np.flip(self.new_data_mask, axis=(0,1))
        self.new_data_vector = np.array([[2],[2],[3],[4],[4]])
        self.new_data_vector_flipped = np.flip(self.new_data_vector, axis=(0,1))
        self.weight_importance = np.array([[0,0,0,1,1,1,1],
                                           [0,0,0,1,1,1,1],
                                           [0,0,1,0,1,1,1],
                                           [0,0,0,1,1,1,1],
                                           [0,0,0,1,1,1,1]])
        self.age_max = env_config['age_max']
        
        
        ''' action space '''
        self.radar_directions_n = 4
        self.data_directions_n = 4
        # communication direction X sensor direction + 1 null action + 1 radar action
        self.action_space = spaces.Discrete(self.radar_directions_n * self.data_directions_n + 2)
        
        self.radar_angular_range = env_config['radar_angular_range']   # beam 40 X 2 degrees wide
        self.radar_range = 100          # 100 m range
        
        ''' observation space '''
        self.data_map_shape = self.new_data_mask.shape
        
        if self.ob_time:
            self.high = np.concatenate((np.ones((1,))*(self.N*4), # num vehicles * radar interval
                                        np.ones((1,)),               # in map
                                        np.ones((1,)) * self.v,               # velocity
                                        self.max_x_dim * np.ones((2*(self.N_obs-1), )),
                                        self.v * np.ones((2*(self.N_obs-1), )),
                                        (self.age_max + 1) * np.ones( ((2 * self.data_map_shape[0] * self.data_map_shape[1]), ))
                                        ), axis=0)
        else:
            self.high = np.concatenate((np.ones((1,)),
                                        np.ones((1,)) * self.v,               # velocity
                                        self.max_x_dim * np.ones((2*(self.N_obs-1), )),
                                        self.v * np.ones((2*(self.N_obs-1), )),
                                        (self.age_max + 1) * np.ones( ((2 * self.data_map_shape[0] * self.data_map_shape[1]), ))
                                        ), axis=0)
        self.observation_space = spaces.Box(low=0, high=self.high, shape = (self.high.shape[0],))
        
        
    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]
    
    def state_transition(self, state, f_sent, f_num_transmits):
        
        state = {}
        
        # update position
        for n in range(self.N):
            # If vehicle n is in the map
            
            if (0 <= self.positions[n,0] <= self.max_x_dim):
                delta_position = self.velocities[(n)] * self.timestep
                self.positions[n] = self.positions[n] + (delta_position)
                if not (0 <= self.positions[n,0] <= self.max_x_dim):
                    self.remove_vehicle(n)
                
                # add received data
                #new_data = newer data + new data
                received_data = (self.data_age[(n+1)] > f_sent[n]) * (f_sent[n]>0) + (f_sent[n]>0) * (self.data_age[(n+1)] == 0)
                self.data_transmits[(n+1)] = (received_data * (f_num_transmits[n] + 1)) + (np.invert(received_data) * self.data_transmits[(n+1)])
                self.data_age[(n+1)] = (received_data * f_sent[n]) + (np.invert(received_data) * self.data_age[(n+1)])
                self.data[(n+1)] = (received_data * self.received_mask) + (np.invert(received_data) * self.data[(n+1)])
                
                # shift data according to new position
                shift_x = (delta_position[0] / self.x_width).astype(int)
                shift_y = (delta_position[1] / self.y_width).astype(int)
                
                if shift_x > 0:
                    self.data[(n+1)] = np.pad(self.data[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    self.data_age[(n+1)] = np.pad(self.data_age[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    self.data_transmits[(n+1)] = np.pad(self.data_transmits[(n+1)], ((0,0), (0, shift_x)), mode='constant')[:,shift_x:]
                    
                    # update with new data
                    new_data = np.tile(self.new_data_vector, shift_x-1)
                    new_data = np.insert(self.new_data_mask, [self.new_data_mask.shape[0]//2 + 1] * shift_x, new_data, axis=1)[:,-self.new_data_mask.shape[1]:]
                
                if shift_x < 0:
                    self.data[(n+1)] = np.pad(self.data[(n+1)], ((0,0), (abs(shift_x), 0)), mode='constant')[:,:shift_x]
                    self.data_age[(n+1)] = np.pad(self.data_age[(n+1)], ((0,0), (abs(shift_x), 0)), mode='constant')[:,:shift_x]
                    
                    # update with new data
                    new_data = np.tile(self.new_data_vector_flipped, abs(shift_x)-1)
                    new_data = np.insert(self.new_data_mask_flipped, [self.new_data_mask.shape[0]//2 + 2] * shift_x, new_data, axis=1)[:,-self.new_data_mask.shape[1]:]
                
                # increment data age
                self.data_age[(n+1)] = self.data_age[(n+1)] + (self.data_age[(n+1)] > 0)
                self.data[(n+1)][self.data_age[(n+1)] > self.age_max] = 0
                self.data_age[(n+1)][self.data_age[(n+1)] > self.age_max] = 0
                # add new data
                self.data[(n+1)] = new_data + (self.data[(n+1)] * (new_data == 0))
                self.data_age[(n+1)] = (self.data_age[(n+1)] * (new_data == 0)) + (new_data > 0)
                self.data_transmits[(n+1)] = (self.data_transmits[(n+1)] * (new_data == 0))
                
            else:
                self.remove_vehicle(n)
            
            state[(n+1)] = self._get_obs(n)
            
        idx_out = np.where(self.in_map == 0)[0]
        self.new_veh_queue += np.random.poisson(self.veh_enter_rate)
        
        if (len(idx_out) > 0) & (self.new_veh_queue > 0) :
            new_idx = np.random.choice(idx_out)
            self.reset_vehicle(new_idx)
            self.new_veh_queue -= 1
            state[new_idx+1] = self._get_obs(new_idx)
        
        return state
        
        
    def get_reward(self, state, action):
        relative_distance = np.zeros((self.N, self.N))
        relative_positions = np.zeros((self.N, self.N, 2))
        targeted = np.zeros((self.N, self.N))
        targeted_front = np.zeros((self.N, self.N))
        targeted_back = np.zeros((self.N, self.N))
        targeted_left = np.zeros((self.N, self.N))
        targeted_right = np.zeros((self.N, self.N))
        radar_targeted = np.zeros((self.N, self.N))
        noise_power = np.zeros((self.N, self.N))
        f_sends = np.zeros((self.N, self.N, self.new_data_mask.shape[0], self.new_data_mask.shape[1]))
        r_rad = np.zeros((self.N,))
        f_num_transmits = np.zeros((self.N, self.new_data_mask.shape[0], self.new_data_mask.shape[1]))
        
        # find vehicles targeted
        for i in range(self.N):
            if self.in_map[i] == 0:
                r_rad[i] = 0
                continue
            
            if action[(i+1)] == (self.action_space.n - 2):
                ac = 'null'
                self.beam_ac[i] = 4
            elif action[(i+1)] == (self.action_space.n - 1):
                ac = 'radar'
                radar_targeted[i] = np.ones((1,self.N))
                radar_targeted[i][i] = 0
            else:
                ac = np.unravel_index(action[(i+1)], (self.radar_directions_n, self.data_directions_n))
                self.beam_ac[i] = ac[0]
            
            r_rad[i] = np.exp(np.abs(self.velocities[i,0]*.05)) if ac != 'radar' else 0
            
            # Euclidean distance between ego vehicle and other vehicles
            relative_distance[i] = np.linalg.norm(self.positions - self.positions[i], axis=1)
            relative_positions[i] = self.positions - self.positions[i]
            
            if (ac != 'null') and (ac != 'radar'):
                ac_vector = rotation(self.velocities[i], ac[0])
                f_send = (self.data[(i+1)] == (ac[1] + 1)) * np.clip((-self.data_age[(i+1)] + self.age_max), 0, self.age_max)
                f_num_transmits[i] = (self.data[(i+1)] == (ac[1] + 1)) * self.data_transmits[(i+1)]
                

                targeted[i] = (abs(vector_angle(ac_vector, relative_positions[i])) < self.radar_angular_range) * self.in_map
                targeted[i][i] = 0
                
                incident_angle = vector_angle(ac_vector, self.velocities)             # angle between beam and velocities
                targeted_front[i] = (abs(incident_angle) < self.radar_angular_range) * targeted[i]
                targeted_back[i] = (abs(incident_angle) > (90+self.radar_angular_range)) * targeted[i]
                targeted_left[i] = ((incident_angle >= 45) & (incident_angle <= (90+self.radar_angular_range))) * targeted[i]
                targeted_right[i] = ((incident_angle <= -45) & (incident_angle >= -(90+self.radar_angular_range))) * targeted[i]
                
                # number of units in x dimension each other vehicle is away from ego vehicle n
                x_unit_distance = np.round(relative_positions[i,:,0] / self.x_width).astype(int)
                y_unit_distance = np.round(relative_positions[i,:,1] / self.y_width).astype(int)
                
                for j in range(self.N):
                    if targeted[i][j] > 0:
                        if x_unit_distance[j] > 0:
                            f_sends[i][j] = np.pad(f_send, ((0,0), (x_unit_distance[j], 0)), mode='constant')[:, 0:-x_unit_distance[j]]
                        else:
                            shift = np.abs(x_unit_distance[j])
                            f_sends[i][j] = np.pad(f_send, ((0,0), (0, shift)), mode='constant')[:, shift:]
                        if y_unit_distance[j] > 0:
                            f_sends[i][j] = np.pad(f_sends[i][j], ((y_unit_distance[j],0), (0, 0)), mode='constant')[0:-y_unit_distance[j], :]
                        else:
                            shift = np.abs(y_unit_distance[j])
                            f_sends[i][j] = np.pad(f_sends[i][j], ((0, shift), (0, 0)), mode='constant')[shift:, :]
        
        power_received_front = power_received(targeted_front, relative_distance)
        power_received_back = power_received(targeted_back, relative_distance)
        power_received_left = power_received(targeted_left, relative_distance)
        power_received_right = power_received(targeted_right, relative_distance)
        
        SINR_front = power_received_front / (noise_power +1e-7)
        SINR_back = power_received_back / (noise_power + 1e-7)
        SINR_left = power_received_left / (noise_power + 1e-7)
        SINR_right = power_received_right / (noise_power + 1e-7)
        
        success_rate_front = success_rate(SINR_front)
        success_rate_back = success_rate(SINR_back)
        success_rate_left = success_rate(SINR_left)
        success_rate_right = success_rate(SINR_right)
        combined_success_rate = success_rate_front + success_rate_back + success_rate_left + success_rate_right
        
        # Log SINR and throughput
        SINR_combined = np.sum(SINR_front + SINR_back + SINR_left + SINR_right, axis=0)
        self.episode_observation['SINR1'] += SINR_combined[0]
        self.episode_observation['SINR_av'] += np.average(SINR_combined)
        
        throughput = np.sum(np.sum(f_sends, axis=(2,3)) * combined_success_rate, axis=1)
        self.episode_observation['throughput1'] += throughput[0]
        self.episode_observation['throughput_av'] += np.average(throughput)
        self.episode_observation['num_transmits1'] += np.sum(f_num_transmits[0])
        self.episode_observation['num_transmits_av'] += np.sum(f_num_transmits) / self.N
        
        # Data transmitted
        f_sent = np.sum(f_sends, axis=(0))
        
        # Compute rewards
        r_comm = np.sum(np.sum(self.weight_importance * f_sends, axis=(2,3)) * combined_success_rate, axis=1)
        
        r = (self.w_comm * r_comm) - (self.w_rad * r_rad)
        
        # Log rewards
        self.episode_observation['r_comm1'] += (r_comm[0])
        self.episode_observation['r_rad1'] += (r_rad[0])
        
        return r, f_sent, f_num_transmits
    
    def heuristic_nn_action(self, n):
        if self.in_map[n] == True:
            ac = np.zeros((2,),dtype=int)
            targeted = np.zeros((self.N,))
            relative_distance = np.linalg.norm(self.positions - self.positions[n], axis=1)
            relative_positions = (self.positions - self.positions[n]) / np.array([self.max_x_dim, self.max_y_dim])
            
            # if other vehicle is in exactly the same position, may be targeted in multiple directions. update targeted only if not previously targeted.
            for d in range(4):
                ac_vector = rotation(self.velocities[n], d)
                targeted = targeted + (targeted==0)*(abs(vector_angle(ac_vector, relative_positions)) < self.radar_angular_range) * self.in_map * (d +1)
            
            mask = np.zeros((self.N,))
            mask[n] = True
            mask[targeted==0] = True    # mask vehicles if they are not targeted
            targeted_vehicle = np.argmin(np.ma.array(relative_distance, mask=mask))
            
            # If no vehicles are targeted, choose null action
            if np.sum(mask) == self.N:
                heuristic_ac = self.action_space.n - 1
            else:
                ac[0] = ac[1] = targeted[targeted_vehicle] - 1
                assert(ac[0] >= 0)
                assert(ac[0] < self.radar_directions_n)
                
                heuristic_ac = np.ravel_multi_index(ac, (self.radar_directions_n, self.data_directions_n))
        
        else:
            heuristic_ac = (self.action_space.n - 1)
        
        return heuristic_ac
        
    def step(self, action):
        r, f_sent, f_num_transmits = self.get_reward(self.state, action)
        r = dict(enumerate(r, 1))
        
        next_state = self.state.copy()
        next_state = self.state_transition(next_state, f_sent, f_num_transmits)
        self.state = next_state
        
        self.episode_observation['step_counter'] += 1
        if self.episode_observation['step_counter'] == 400:
            done = True
            print('End of episode')
        else:
            done = False
        
        return self.state, r, done, {}
        
    def reset(self):
        self.episode_observation = {
            'step_counter': 0,
            'throughput1': 0,
            'throughput_av': 0,
            'SINR1': 0,
            'SINR_av': 0,
            'r_comm1': 0,
            'r_rad1': 0,
            'num_transmits1': 0,
            'num_transmits_av': 0,
        }
        
        """ Generate RSU positions """
        x_pos_RSU = np.random.choice(self.max_x_dim , size=self.N_RSU, replace = False)
        self.RSU_positions = x_pos_RSU
        
        """ Generate vehicle positions 
        - position_idx generates places without replacement from dimensions (road length) x (num lanes)
        """
        position_idx = np.random.choice(self.max_x_dim * self.num_lanes, size=self.N, replace = False)
        x_position = position_idx - position_idx//self.max_x_dim * self.max_x_dim
        lane = (position_idx // self.max_x_dim)
        y_position = ((lane - self.num_lanes//2 + 0.5) * self.lane_width)
        lane_num = lane - self.num_lanes//2
        lane_num[lane_num<0] = lane_num[lane_num<0] + 1
        
        self.positions = np.stack((x_position, y_position)).transpose().astype('float32')
        self.velocities = np.zeros_like(self.positions)
        self.velocities[:,0] = self.v * np.sign(self.positions[:,1]) + lane_num * self.v_diff
        self.in_map = np.ones((self.N,))
        self.beam_ac = 5*np.ones((self.N,))
        
        self.data, self.data_age, state = {}, {}, {}
        self.data_transmits = {}
        for n in range(self.N):
            self.data[(n+1)] = (lane[n]==0) * self.new_data_mask_flipped + (lane[n]==1)*self.new_data_mask
            self.data_age[(n+1)] = (self.data[(n+1)] > 0) * 1
            self.data_transmits[(n+1)] = np.zeros_like(self.data[(n+1)])
            state[(n+1)] = self._get_obs(n)
        
        self.state = state.copy()
        return state
    
    def remove_vehicle(self, n):
        self.in_map[n] = 0
        self.data[(n+1)] = np.zeros_like(self.new_data_mask)
        self.data_age[(n+1)] = np.zeros_like(self.new_data_mask)
    
    def reset_vehicle(self, n):
        self.in_map[n] = 1
        lane = np.random.randint(self.num_lanes)
        lane_num = lane - self.num_lanes//2
        lane_num = lane_num + 1*(lane_num < 0)
        
        y_position = ((lane - self.num_lanes//2 + 0.5) * self.lane_width)
        x_position = (y_position < 0) *  self.max_x_dim
        self.positions[n] = np.array([x_position, y_position]).astype('float32')
        self.velocities[n,0] = self.v * np.sign(self.positions[n,1]) + lane_num * self.v_diff
        self.data[(n+1)] = (y_position < 0) * np.flip(self.new_data_mask, axis=(0,1)) + (y_position > 0)*self.new_data_mask
        self.data_age[(n+1)] = (self.data[(n+1)] > 0) * 1
        self.data_transmits[(n+1)] = np.zeros_like(self.data[(n+1)])
    
    def _get_obs(self, n):
        # normalise positions and velocities by range
        relative_positions = (np.delete(self.positions, n, axis=0) - self.positions[n]) / np.array([self.max_x_dim, self.max_y_dim])
        relative_velocities = (np.delete(self.velocities, n, axis=0) - self.velocities[n]) / self.v
        
        if self.N_obs < self.N:
            l2_dist = np.linalg.norm(relative_positions, axis=1)
            l2_dist_idx = l2_dist.argsort()
            relative_positions = relative_positions[l2_dist_idx]
            relative_velocities = relative_velocities[l2_dist_idx]
            relative_positions = relative_positions[0:self.N_obs-1]
            relative_velocities = relative_velocities[0:self.N_obs-1]
        elif self.N_obs > self.N:
            zeros = np.zeros((self.N_obs-self.N,2))
            relative_positions = np.concatenate((relative_positions, zeros))
            relative_velocities = np.concatenate((relative_velocities, zeros))
        
        # normalise data by num directions (i.e. 4)
        if self.ob_time:
            state = np.concatenate((np.array(self.episode_observation['step_counter'] % (4 * self.N)).reshape(-1),
                                    self.in_map[n].reshape(-1),
                                    self.velocities[n,0].reshape(-1),
                                    relative_positions.reshape(-1),
                                    relative_velocities.reshape(-1),
                                    (self.data[(n+1)] / 4).reshape(-1),
                                    (self.data_age[(n+1)] / self.age_max).reshape(-1)
                                    ))
        else:
            state = np.concatenate((self.in_map[n].reshape(-1),
                                    self.velocities[n,0].reshape(-1),
                                    relative_positions.reshape(-1),
                                    relative_velocities.reshape(-1),
                                    (self.data[(n+1)] / 4).reshape(-1),
                                    (self.data_age[(n+1)] / self.age_max).reshape(-1)
                                    ))
        state = np.divide(state, self.high)
        state = np.expand_dims(state, axis=0)
        return state
    
    def render(self, mode='human'):
        """
        
        Produces a rendering of the dual carriageway JRC game environment.

        """
        
        screen_width = 600
        screen_height = 400
        
        scale_x = screen_width/self.max_x_dim
        scale_y = screen_height/self.max_y_dim
        carwidth = 40
        carheight = 20
        
        # Geoemetry for beam triangle
        triangle_height = carheight
        triangle_halfwidth = triangle_height * np.tan(30/180*np.pi)
        v1 = np.array([carheight/2,0])
        v2 = v1 + np.array([triangle_height, -triangle_halfwidth])
        v3 = v1 + np.array([triangle_height, triangle_halfwidth,])
        vertices = np.transpose(np.array([v1,v2,v3]))
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            clearance = -carheight/2
            self.cartrans = {}
            
            for n in range(self.N):
                l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
                car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                if n == 0:
                    car.set_color(1,0,0)
                car.add_attr(rendering.Transform(translation=(0, clearance)))
                self.cartrans[n+1] = rendering.Transform()
                car.add_attr(self.cartrans[n+1])
                self.viewer.add_geom(car)
                frontwheel = rendering.make_circle(carheight / 2.5)
                frontwheel.set_color(.5, .5, .5)
                frontwheel.add_attr(
                    rendering.Transform(translation=(carwidth / 4, clearance))
                )
                frontwheel.add_attr(self.cartrans[n+1])
                self.viewer.add_geom(frontwheel)
                backwheel = rendering.make_circle(carheight / 2.5)
                backwheel.add_attr(
                    rendering.Transform(translation=(-carwidth / 4, clearance))
                )
                backwheel.add_attr(self.cartrans[n+1])
                backwheel.set_color(.5, .5, .5)
                self.viewer.add_geom(backwheel)
            
        for n in range(self.N):
            pos = self.positions[n]
            x = pos[0] * scale_x
            y = (pos[1] * scale_y) + (screen_height / 2)
            self.cartrans[n+1].set_translation(x, y)
            
            # draw beams
            if self.beam_ac[n] != 5:
                vertex_direction = vertices * np.sign(self.velocities[n,0])
                v1, v2, v3 = np.transpose(rotation(vertex_direction, self.beam_ac[n])) + np.array([x,y])
                self.viewer.draw_polygon([tuple(v1), tuple(v2), tuple(v3)], color=(0,0,1))
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
                






def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(test)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def alternative_switch_action(t, num_actions):
    """
    Alternates between communication '0' and a choice of communications actions.
    Cycles between the communication actions

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        r = t % (num_actions*2)
        return int(r/2 + 1)

def alt_switch_action5(t, comm_action):
    """
    Alternates between communication '0' and communicating packets with urgency level 'comm_action'.

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        return comm_action

class Agent(object):
    def __init__(self, policy_config, sample_trajectory_args, env_args):
        
        self.num_users = env_args['num_users']
        
        self.mode = policy_config['mode']
        self.ac_dim = policy_config['ac_dim']
        self.CW_min = policy_config['CW'][0]
        self.CW_max = policy_config['CW'][1]
        self.CW = np.ones((self.num_users,)) * self.CW_min
        self.counter = np.zeros((self.num_users,))
        
        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        
        self.timesteps_this_batch = 0
        
        self.counter = np.random.randint(self.CW, size=self.num_users)
    
    
    def act(self, ob):
        # if counter is zero select actions
        actions = (self.counter==0) * np.random.randint(self.ac_dim)
        action_reqs = actions
        
        priorities = np.empty((0),dtype=int)
        for n in range(self.num_users):
            priorities = np.concatenate((priorities, ob[(n+1)][state_space_size['data_size']*2].reshape(1)))
        
        if np.sum(actions>0) > 1:
            actions = (priorities == 1) * action_reqs   # Choose agent that transmits based on priority
        unsuccessful_ac_reqs = (actions==0) * (action_reqs!=0)         # agents that act but unsuccessful
        
        # Halve CW for successful transmission, double for unsuccessful transmission
        self.CW = np.clip(((actions>0) * self.CW / 2) + ((actions==0) * self.CW), 2, self.CW_max)
        self.CW = np.clip((unsuccessful_ac_reqs * self.CW * 2) + (unsuccessful_ac_reqs==0 * self.CW), 2, self.CW_max)
        
        # decrement counter
        self.counter = np.clip(self.counter - 1, a_min=0, a_max=self.CW_max)
        # reset counter for agents that attempted to take action
        self.counter = ((action_reqs>0) * np.random.randint(self.CW, size=self.num_users)) + ((action_reqs==0) * self.counter)
        
        ac = {}
        for n in range(self.num_users):
            ac[(n+1)] = actions[n]
            
        return ac
    
    
    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        self.timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            self.timesteps_this_batch += pathlength(path[1])
            if self.timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return paths, self.timesteps_this_batch
    
    
    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()                    # returns ob['agent_no'] = 
        obs, acs, log_probs, rewards, next_obs, next_acs, hiddens, entropys = {}, {}, {}, {}, {}, {}, {}, {}
        prev_acs = {}
        terminals = []
        for i in range(self.num_users):
            obs[(i+1)], acs[(i+1)], log_probs[(i+1)], rewards[(i+1)], next_obs[(i+1)], next_acs[(i+1)], hiddens[(i+1)], entropys[(i+1)] = \
            [], [], [], [], [], [], [], []
        
        steps = 0
        
        for i in range(self.num_users):
            if self.mode == 'unif_rand':
                acs[(i+1)] = np.array(np.random.randint(env.action_space.n))
            elif self.mode == 'rotate':
                acs[(i+1)] = np.array((steps + i) % env.action_space.n)
            elif self.mode == 'urg5':
                acs[(i+1)] = alt_switch_action5(steps, 5)
            elif self.mode == 'heuristic':
                if i==0:
                    acs[(i+1)] = np.array(env.heuristic_nn_action(i))
                else:
                    acs[(i+1)] = np.array((steps + i) % env.action_space.n)
        if self.mode == 'csma-ca':
            acs = self.act(ob)

        
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            ob, rew, done, _ = env.step(acs)
            
            for i in range(self.num_users):
                rewards[(i+1)].append(rew[(i+1)])     # most recent reward appended to END of list
                
                if self.mode == 'unif_rand':
                    acs[(i+1)] = np.array(np.random.randint(env.action_space.n))
                elif self.mode == 'rotate':
                    acs[(i+1)] = np.array((steps + i) % env.action_space.n)
                elif self.mode == 'urg5':
                    acs[(i+1)] = alt_switch_action5(steps, 5)
                elif self.mode == 'heuristic':
                    if i==0:
                        acs[(i+1)] = np.array(env.heuristic_nn_action(i))
                    else:
                        acs[(i+1)] = np.array((steps + i) % env.action_space.n)
            if self.mode == 'csma-ca':
                acs = self.act(ob)
            
            steps += 1
            if done or steps >= self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {}
        for i in range(self.num_users):
            path[(i+1)] = {"reward" : np.array(rewards[(i+1)], dtype=np.float32),                  # (steps)
                 }
            path[(i+1)]["action"] = np.array(acs[(i+1)], dtype=np.float32)
        
        path["terminal"] = np.array(terminals, dtype=np.float32)
        
        # Log additional statistics
        path['r_comm1'] = (env.episode_observation['r_comm1'])
        path['r_rad1'] = (env.episode_observation['r_rad1'])
        path['throughput1'] = (env.episode_observation['throughput1'] / 400)
        path['throughput_av'] = (env.episode_observation['throughput_av'] / 400)
        path['SINR1'] = (env.episode_observation['SINR1'] / 400)
        path['SINR_av'] = (env.episode_observation['SINR_av'] / 400)
        path['num_transmits1'] = (env.episode_observation['num_transmits1'] / 400)
        path['num_transmits_av'] = (env.episode_observation['num_transmits_av'] / 400)
        
        return path


def test(
        exp_name,
        env_name,
        env_config,
        n_iter, 
        min_timesteps_per_batch, 
        max_path_length,
        animate,
        seed,
        mode,
        CW,
        logdir,
        ):
    
    start = time.time()
    setup_logger(logdir, locals())  # setup logger for results    
    
    env = Beamform_JRC(env_config)
    # if animate == True:
    #     env = Monitor(env, './video', force=True)
    env.seed(seed)
    num_users = int(env.N)
    
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    policy_config = {'mode': args.mode,
                     'CW': CW,
                     'ac_dim': env.action_space.n,
                         }
    env_args = {'num_users': env_config['num_users']}
    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }
    agent = Agent(policy_config, sample_trajectory_args, env_args)
    
    total_timesteps = 0
    
    for itr in range(n_iter):
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
    
        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no, ac_na, re_n, log_prob_na, next_ob_no, next_ac_na, h_ns1, entropy = {}, {}, {}, {}, {}, {}, {}, {}
        returns = np.zeros((num_users,len(paths)))
        for i in range(num_users):
            re_n[(i+1)] = np.concatenate([path[(i+1)]["reward"] for path in paths])               # (batch_size, num_users)
            returns[i,:] = [path[(i+1)]["reward"].sum(dtype=np.float32) for path in paths]   # (num_users, num episodes in batch)
            assert re_n[(i+1)].shape == (timesteps_this_batch,)
            assert returns[i,:].shape == (timesteps_this_batch/400,)
                
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        # Log additional statistics
        r_comm1 = ([path["r_comm1"] for path in paths])  
        r_rad1 = ([path["r_rad1"] for path in paths])   
        throughput1 = ([path["throughput1"] for path in paths])                      # (batch,)
        throughput_av = ([path["throughput_av"] for path in paths])
        SINR1 = ([path["SINR1"] for path in paths])                      # (batch,)
        SINR_av = ([path["SINR_av"] for path in paths])
        num_transmits1 = ([path["num_transmits1"] for path in paths])
        num_transmits_av = ([path["num_transmits_av"] for path in paths])
        
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Average Reward", np.mean(returns))   # per agent per episode
        logz.log_tabular("StdReward", np.std(returns))
        logz.log_tabular("MaxReward", np.max(returns))
        logz.log_tabular("MinReward", np.min(returns))
        # logz.log_tabular("throughput1", np.mean(throughput1))
        # logz.log_tabular("throughput_av", np.mean(throughput_av))
        logz.log_tabular("r_comm1", np.mean(r_comm1))
        logz.log_tabular("r_rad1", np.mean(r_rad1))
        logz.log_tabular("Throughput1", np.mean(throughput1))
        logz.log_tabular("Average Throughput", np.mean(throughput_av))
        logz.log_tabular("SINR1", np.mean(SINR1))
        logz.log_tabular("SINR_av", np.mean(SINR_av))
        logz.log_tabular("Num Transmits 1", np.mean(num_transmits1))
        logz.log_tabular("Average Num Transmits", np.mean(num_transmits_av))
        
        for i in range(env_config['num_users']):
            logz.log_tabular("Reward"+str(i+1), np.mean(returns, axis=1)[i])
            logz.log_tabular("StdReward"+str(i+1), np.std(returns, axis=1)[i])
            logz.log_tabular("MaxReward"+str(i+1), np.max(returns, axis=1)[i])
            logz.log_tabular("MinReward"+str(i+1), np.min(returns, axis=1)[i])
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        
        logz.dump_tabular(step=itr)

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, default='beamform_JRC-v0')
parser.add_argument('--num_lanes', type=int, default=2)
parser.add_argument('--num_users_NN', type=int)
parser.add_argument('--num_users', type=int, nargs='+', default = [0])
parser.add_argument('--num_agents', type=int)
parser.add_argument('--x_dim', type=int, nargs='+', default=[150, 150, 10])
parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
parser.add_argument('--obj', choices=['peak','avg'], default='avg')
parser.add_argument('--age_max', type=float, default=3)
parser.add_argument('--w_comm', type=float, default=1)
parser.add_argument('--w_rad', type=float, default=1)
parser.add_argument('--ang_range', type=float, default=45)

# Algorithm hyperparameters
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_experiments', '-e', type=int, default=1)
parser.add_argument('--mode', choices=['heuristic','urg5','best','csma-ca','rotate'], default='rotate')
parser.add_argument('--CW', type=int, nargs='+', default=[2,16])

parser.add_argument('--exp_name', type=str, default='vpg')
parser.add_argument('--render', action='store_true')

parser.add_argument('--n_iter', '-n', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=1000)
args = parser.parse_args()


logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join('data', logdir)

print("------")
print(logdir)
print("------")

max_path_length = args.ep_len if args.ep_len > 0 else None

if args.num_users == [0]:
        args.num_users[0] = args.num_users_NN

for num_users in args.num_users:
    for x_dim in range(args.x_dim[0], args.x_dim[1], args.x_dim[2]):
                
        for e in range(args.n_experiments):
            """ Set random seeds 
            https://keras.io/getting_started/faq/
            """
            seed = args.seed + e*10
            # The below is necessary for starting Numpy and Python generated random numbers in a well-defined initial state.
            np.random.seed(seed)
            python_random.seed(seed)
            
            env_config = {'num_users': num_users,
                          'num_users_NN': args.num_users_NN,
                          'num_agents': args.num_agents,
                          'age_max': args.age_max,
                          'x_dim': x_dim,
                          'num_lanes': args.num_lanes,
                          'w_comm': args.w_comm,
                          'w_rad': args.w_rad,
                          'radar_angular_range': args.ang_range,
                          }
            
            logdir_w_params = logdir + "_{}usrs_{}_wcomm1{}_w_rad{}_maxage_{}_ang{}".format(num_users,args.mode,args.w_comm, args.w_rad, args.age_max,args.ang_range)
            
            test(
                exp_name = args.exp_name,
                env_name = args.env_name,
                env_config = env_config,
                n_iter = args.n_iter, 
                min_timesteps_per_batch = args.batch_size, 
                max_path_length = max_path_length,
                animate = args.render,
                seed = args.seed,
                mode = args.mode,
                CW = args.CW,
                logdir = os.path.join(logdir_w_params,'%d'%seed),
                )
                        
