import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.stats import qmc
import pandas as pd
from collections import defaultdict
from abc import abstractmethod
from pcgym import make_env


np.random.seed(42)

@dataclass
class BaseModel: # As inferred from complex_cstr
    int_method: str = 'jax'
    states: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    disturbances: List[str] = field(default_factory=list)
    uncertainties: Dict[str, float] = field(default_factory=dict)

    @abstractmethod
    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def info(self) -> dict:
        pass

@dataclass
class SimulationResult:
    """Container for simulation results."""
    observed_states: Dict[str, List[float]]
    actions: Dict[str, List[float]]

    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        """Makes SimulationResult iterable, yielding (observed_states, disturbances, actions)"""
        return iter((self.observed_states, self.actions))

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int  # Number of time steps
    tsim: int # Simulation time period (used for generating changes within T)
    noise_percentage: Union[float, Dict] = 0.01

class SimulationConverter():
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert output of simulation and return features and targets"""
        pass

class DistillationColumnConverter(SimulationConverter):
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        obs_states_list = [res.observed_states for res in data]
        disturbances_list = [res.disturbances for res in data]
        actions_list = [res.actions for res in data]

        combined_obs_states = defaultdict(list)
        for obs_series in obs_states_list:
            for key, values in obs_series.items():
                combined_obs_states[key].append(values)
        
        combined_disturbances = defaultdict(list)
        for dist_series in disturbances_list:
            for key, values in dist_series.items():
                combined_disturbances[key].append(values)

        combined_actions = defaultdict(list)
        for act_series in actions_list:
            for key, values in act_series.items():
                combined_actions[key].append(values)

        for key in combined_obs_states: combined_obs_states[key] = np.array(combined_obs_states[key])
        for key in combined_disturbances: combined_disturbances[key] = np.array(combined_disturbances[key])
        for key in combined_actions: combined_actions[key] = np.array(combined_actions[key])
            
        feature_keys_obs = ["x_D", "x_1", "x_B"] # From model_instance.states
        feature_keys_dist = ["F_feed", "X_feed", "q_feed"] # From model_instance.disturbances
        feature_keys_act = ["L_R", "V_B"] # From model_instance.inputs

        all_features_dict = {}
        for key in feature_keys_obs: all_features_dict[key] = combined_obs_states[key]
        for key in feature_keys_dist: all_features_dict[key] = combined_disturbances[key]
        for key in feature_keys_act: all_features_dict[key] = combined_actions[key]
            
        num_simulations = len(data)
        timesteps_per_simulation = len(data[0].observed_states["x_D"])
        sim_no_array = np.array([[i + 1] * timesteps_per_simulation for i in range(num_simulations)])
        all_features_dict['simulation_no'] = sim_no_array

        ordered_feature_keys = feature_keys_obs + feature_keys_dist + feature_keys_act + ['simulation_no']
        features_stacked = np.stack([all_features_dict[key] for key in ordered_feature_keys], axis=-1)
        features = features_stacked.reshape(-1, features_stacked.shape[-1])

        target_keys = ["x_D", "x_1", "x_B"] 
        targets_stacked = np.stack([combined_obs_states[key] for key in target_keys], axis=-1)
        targets = targets_stacked.reshape(-1, targets_stacked.shape[-1])
        
        return features, targets


class DistillationColumnSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config

        # Inputs: L_R (Reflux flow), V_B (Boil-up rate)
        self.action_space = { # Adjust ranges based on column design, F_param, D_param
            'low': np.array([10, 2]), # L_R_min, V_B_min
            'high': np.array([100, 10])  # L_R_max, V_B_max
        }
        # States: x_D, x_1, x_B (Compositions 0-1)
        self.observation_space = {
            'low': np.array([0.0] * 9),
            'high': np.array([1.0] * 9)
        }
        self.num_states = len(self.observation_space['low'])
        self.num_actions = len(self.action_space['low'])





    def generate_lhs_actions(self) -> np.ndarray: # Returns (T, num_actions)
        sampler = qmc.LatinHypercube(d=self.num_actions)
        max_action_changes = self.config.T // 4
        samples = sampler.random(n=max_action_changes)
        action_sequences = np.zeros((self.config.T, self.num_actions))

        for i in range(self.num_actions):
            act_low = self.action_space['low'][i]
            act_high = self.action_space['high'][i]
            act_range = act_high - act_low
            current_samples = act_low + samples[:, i] * act_range
            
            num_changes = np.random.randint(0, max_action_changes)
            change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
            
            sample_idx = 0
            current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05

            for t in range(self.config.T):
                if len(change_points) > 0 and t == change_points[0]:
                    sample_idx = (sample_idx + 1) % len(current_samples)
                    current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05
                    change_points = change_points[1:]
                action_sequences[t, i] = current_action_val
        return action_sequences

    def generate_x0(self) -> np.ndarray:
        """Generate initial state for the simulation as a random deviation from the midpoint of each observation bound."""
        midpoints = (self.observation_space['high'] + self.observation_space['low']) / 2
        deviations = (np.random.rand(*self.observation_space['low'].shape) - 0.5) * (self.observation_space['high'] - self.observation_space['low']) * 0.1
        return midpoints + deviations
    
    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        unnorm_action_sequence = self.generate_lhs_actions()
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence)
        x0 = self.generate_x0()
        
        env_params = {
            'N': self.config.T, 
            'tsim': self.config.tsim, 
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0,
            'reward_states': np.array(['X0']),
            'maximise_reward': True,
            'model': 'distillation_column', 
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'integration_method': 'jax',
        }
        
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        sim_obs_states, sim_actions = [], []
        nl_obs_states = []

        obs_norm, _ = env.reset()
        nl_obs_norm, _ = noiseless_env.reset()
        
        done = False
        step = 0
        while not done:
            action_norm = action_sequence_normalized[step, :]
            obs_norm, _, done, _, _ = env.step(action_norm)
            nl_obs_norm, _, _, _, _ = noiseless_env.step(action_norm)

            current_obs_norm = obs_norm[:self.num_states]
            nl_current_obs_norm = nl_obs_norm[:self.num_states]

            obs_unnorm = self._unnormalize_observation(current_obs_norm)
            action_unnorm_step = self._unnormalize_action(action_norm)
            nl_obs_unnorm = self._unnormalize_observation(nl_current_obs_norm)
           
            sim_obs_states.append(obs_unnorm)
            sim_actions.append(action_unnorm_step)
            nl_obs_states.append(nl_obs_unnorm)
            
            step += 1
            if step >= self.config.T: done = True
        
        actions_array = np.array(sim_actions)
        return (
            self._format_results(sim_obs_states, actions_array),
            self._format_results(nl_obs_states, actions_array)
        )

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        noisy_res, noiseless_res = [], []
        for _ in tqdm(range(self.config.n_simulations), desc="Running Distillation Sims"):
            n_sim, nl_sim = self.simulate()
            noisy_res.append(n_sim)
            noiseless_res.append(nl_sim)
        return noisy_res, noiseless_res

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        # Similar plotting logic as MultistageExtractionSimulator, adapt titles/labels
        num_sims = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sims))

        state_names = ['X0', 'X1', 'X2', 'X3', 'Xf', 'X4', 'X5', 'X6', 'Xb']
        action_names = ['R', 'F']

        # Changed to 5,2 subplot configuration
        fig_states, axs_states = plt.subplots(5, 2, figsize=(15, 25))
        # Flatten for easier indexing
        axs_states_flat = axs_states.flatten()
        
        for i in range(num_sims):
            noisy_res, noiseless_res = noisy_results[i], noiseless_results[i]
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""
            for j, state_name in enumerate(state_names):
                axs_states_flat[j].plot(noisy_res.observed_states[state_name], color=color, alpha=0.8, label=f'Noisy{label_suffix}')
                axs_states_flat[j].plot(noiseless_res.observed_states[state_name], color=color, linestyle='--', alpha=0.8, label=f'Noiseless{label_suffix}')
                axs_states_flat[j].set_title(f'State: {state_name}')
                axs_states_flat[j].set_xlabel('Time')
                axs_states_flat[j].set_ylabel('Composition')
        
        # Remove the extra subplot
        if len(axs_states_flat) > len(state_names):
            fig_states.delaxes(axs_states_flat[-1])
            
        if num_sims > 1:
            for j in range(len(state_names)): 
                axs_states_flat[j].legend(loc='best', fontsize='small')
        
        fig_states.tight_layout()
        plt.show()

        fig_actions, axs_actions = plt.subplots(len(action_names), 1, figsize=(12, len(action_names) * 3), squeeze=False)
        for i in range(num_sims):
            noisy_res = noisy_results[i]
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""
            for j, act_name in enumerate(action_names):
                axs_actions[j,0].plot(noisy_res.actions[act_name], color=color, label=f'{act_name}{label_suffix}')
                axs_actions[j,0].set_title(f'Action: {act_name}')
                axs_actions[j,0].set_xlabel('Time')
        if num_sims > 1:
            for j in range(len(action_names)): axs_actions[j,0].legend(loc='best', fontsize='small')
        fig_actions.tight_layout()
        plt.show()
        
    def _normalize_action(self, action_values: np.ndarray) -> np.ndarray:
        # action_values can be (T, num_actions) or (num_actions,)
        low = self.action_space['low']
        high = self.action_space['high']
        return 2 * (action_values - low) / (high - low) - 1

    def _unnormalize_action(self, norm_actions: np.ndarray) -> np.ndarray:
        low = self.action_space['low']
        high = self.action_space['high']
        return (norm_actions + 1) * (high - low) / 2 + low

    def _normalize_observation(self, obs_values: np.ndarray) -> np.ndarray:
        low = self.observation_space['low']
        high = self.observation_space['high']
        return 2 * (obs_values - low) / (high - low) - 1

    def _unnormalize_observation(self, norm_obs: np.ndarray) -> np.ndarray:
        low = self.observation_space['low']
        high = self.observation_space['high']
        return (norm_obs + 1) * (high - low) / 2 + low


    def _format_results(
        self,
        observed_states_list: List[np.ndarray], # List of state vectors [X1,..Y5] per time step   # List of disturbance vectors [X0f, Y6f] per time step
        actions_array: np.ndarray              # Array of action vectors (T, num_actions) [L, G]
    ) -> SimulationResult:
        
        num_timesteps = len(observed_states_list)
        
        obs_states_dict = {'X0': [state[0] for state in observed_states_list],
                            'X1': [state[1] for state in observed_states_list],
                            'X2': [state[2] for state in observed_states_list],
                            'X3': [state[3] for state in observed_states_list],
                            'Xf': [state[4] for state in observed_states_list],
                            'X4': [state[5] for state in observed_states_list],
                            'X5': [state[6] for state in observed_states_list],
                            'X6': [state[7] for state in observed_states_list],
                            'Xb': [state[8] for state in observed_states_list],
                            }

        # actions_array is already (T, num_actions)
        action_states_dict = {'R': [state[0] for state in actions_array],
                              'F': [state[1] for state in actions_array]}
        
        return SimulationResult(obs_states_dict, action_states_dict)
# --- Example Usage for Distillation Column ---
config_dist = SimulationConfig(n_simulations=10, T=100, tsim=100, noise_percentage=0.01)
simulator_dist = DistillationColumnSimulator(config_dist)

noisy_results_dist, noiseless_results_dist = simulator_dist.run_multiple_simulations()
simulator_dist.plot_results(noisy_results_dist, noiseless_results_dist)

# converter_dist = DistillationColumnConverter() # Uses the model_instance from simulator for names
# features_dist, targets_dist = converter_dist.convert(noisy_results_dist)

# print("Distillation Column Features Shape:", features_dist.shape)
# print("Distillation Column Targets Shape:", targets_dist.shape)

# features_df_dist = pd.DataFrame(features_dist, columns=simulator_dist.model_instance.states + simulator_dist.model_instance.disturbances + simulator_dist.model_instance.inputs + ['simulation_no'])
# target_names_dist = simulator_dist.model_instance.states
# targets_df_dist = pd.DataFrame(targets_dist, columns=target_names_dist)

# # features_df_dist.to_csv('distillation_column_features.csv', index=False)
# # targets_df_dist.to_csv('distillation_column_targets.csv', index=False)