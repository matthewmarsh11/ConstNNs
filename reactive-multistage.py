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


@dataclass
class SimulationResult:
    """Container for simulation results."""
    observed_states: Dict[str, List[float]]
    disturbances: Dict[str, List[float]]
    actions: Dict[str, List[float]]

    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        """Makes SimulationResult iterable, yielding (observed_states, disturbances, actions)"""
        return iter((self.observed_states, self.disturbances, self.actions))

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
    

class MultistageExtractionReactiveConverter(SimulationConverter):
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        obs_states_list = [res.observed_states for res in data]
        disturbances_list = [res.disturbances for res in data]
        actions_list = [res.actions for res in data]

        # Combine data from all simulations
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

        # Convert lists of lists to numpy arrays (simulations x timesteps)
        for key in combined_obs_states:
            combined_obs_states[key] = np.array(combined_obs_states[key])
        for key in combined_disturbances:
            combined_disturbances[key] = np.array(combined_disturbances[key])
        for key in combined_actions:
            combined_actions[key] = np.array(combined_actions[key])
            
        # Define features: all observed states, disturbances, and actions
        feature_keys_obs = ["XA1", "YA1", "YB1", "YC1", "XA2", "YA2", "YB2", "YC2", 
                           "XA3", "YA3", "YB3", "YC3", "XA4", "YA4", "YB4", "YC4", 
                           "XA5", "YA5", "YB5", "YC5"]
        feature_keys_dist = ["XA0", "YA6", "YB6", "YC6"]
        feature_keys_act = ["L", "G"]

        all_features_dict = {}
        for key in feature_keys_obs:
            all_features_dict[key] = combined_obs_states[key]
        for key in feature_keys_dist:
            all_features_dict[key] = combined_disturbances[key]
        for key in feature_keys_act:
            all_features_dict[key] = combined_actions[key]
            
        # Add simulation number
        num_simulations = len(data)
        timesteps_per_simulation = len(data[0].observed_states["XA1"]) # Assuming all states have same length
        
        # Simulation numbers: shape (num_simulations, timesteps_per_simulation)
        sim_no_array = np.array([[i + 1] * timesteps_per_simulation for i in range(num_simulations)])
        all_features_dict['simulation_no'] = sim_no_array

        # Stack features: result shape (num_simulations, timesteps, num_features)
        # Order of features matters for the final array
        ordered_feature_keys = feature_keys_obs + feature_keys_dist + feature_keys_act + ['simulation_no']
        features_stacked = np.stack([all_features_dict[key] for key in ordered_feature_keys], axis=-1)
        
        # Reshape to (total_samples, num_features)
        features = features_stacked.reshape(-1, features_stacked.shape[-1])

        # Define targets: using observed states as targets
        target_keys = feature_keys_obs
        
        targets_stacked = np.stack([combined_obs_states[key] for key in target_keys], axis=-1)
        targets = targets_stacked.reshape(-1, targets_stacked.shape[-1])
        
        return features, targets

class MultistageExtractionReactiveSimulator:
    def __init__(self, config: SimulationConfig, model_params: Optional[Dict] = None):
        self.config = config

        # Define spaces for the reactive model
        # Inputs: L (Liquid flowrate), G (Gas flowrate)
        self.action_space = {
            'low': np.array([1, 1]),    # L_min, G_min
            'high': np.array([5, 5])    # L_max, G_max
        }
        
        # States for reactive model: 20 states [XA1, YA1, YB1, YC1, XA2, YA2, YB2, YC2, ...]
        self.observation_space = {
            'low': np.array([0.0] * 20),
            'high': np.array([1.0] * 20)
        }
        
        # Disturbances: XA0, YA6, YB6, YC6
        self.disturbance_definition = {
            'XA0': {'low': 0.3, 'high': 0.9, 'idx': 0},
            'YA6': {'low': 0.01, 'high': 0.2, 'idx': 1},
            'YB6': {'low': 0.01, 'high': 0.2, 'idx': 2},
            'YC6': {'low': 0.01, 'high': 0.2, 'idx': 3}
        }
        
        # For internal _unnormalize_disturbance and env_params['disturbance_bounds']
        self.disturbance_space_env = {
            'low': np.array([self.disturbance_definition[d]['low'] for d in ['XA0', 'YA6', 'YB6', 'YC6']]),
            'high': np.array([self.disturbance_definition[d]['high'] for d in ['XA0', 'YA6', 'YB6', 'YC6']])
        }

        self.num_actions = len(self.action_space['low'])
        self.num_states = len(self.observation_space['low'])
        self.num_disturbances = len(self.disturbance_definition)

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for a target state."""
        num_changes = np.random.randint(0, self.config.T // 10)
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
        
        # Using XA5 as an example setpoint (could be any state)
        setpoints_XA5 = []
        sp_low, sp_high = self.observation_space['low'][16]*1.1, self.observation_space['high'][16]*0.9  # XA5 is at index 16
        current_setpoint = np.random.uniform(sp_low, sp_high) if sp_low < sp_high else (sp_low+sp_high)/2
        
        for t in range(self.config.T):
            if len(change_points) > 0 and t == change_points[0]:
                current_setpoint = np.random.uniform(sp_low, sp_high) if sp_low < sp_high else (sp_low+sp_high)/2
                change_points = change_points[1:]
            setpoints_XA5.append(current_setpoint)
        
        return {'XA5_s': setpoints_XA5}

    def generate_disturbances(self) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
        """Generate random disturbances for XA0, YA6, YB6, YC6 using LHS."""
        sampler = qmc.LatinHypercube(d=self.num_disturbances)
        max_dist_changes = self.config.T // 4
        samples = sampler.random(n=max_dist_changes)

        generated_disturbances = {}
        for name, props in self.disturbance_definition.items():
            d_low, d_high, d_idx = props['low'], props['high'], props['idx']
            d_range = d_high - d_low
            d_samples = d_low + samples[:, d_idx] * d_range
            
            num_changes = np.random.randint(0, max_dist_changes)
            change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
            
            current_disturbance_values = []
            sample_idx = 0
            current_val = d_samples[sample_idx] + (np.random.rand() - 0.5) * d_range * 0.05  # Small variation

            for t in range(self.config.T):
                if len(change_points) > 0 and t == change_points[0]:
                    sample_idx = (sample_idx + 1) % len(d_samples)
                    current_val = d_samples[sample_idx] + (np.random.rand() - 0.5) * d_range * 0.05
                    change_points = change_points[1:]
                current_disturbance_values.append(current_val)
            generated_disturbances[name] = current_disturbance_values
            
        return generated_disturbances, self.disturbance_space_env

    def generate_lhs_actions(self) -> np.ndarray:
        """Generate action sequences for L and G using LHS with step changes."""
        sampler = qmc.LatinHypercube(d=self.num_actions)
        max_action_changes = self.config.T // 4
        samples = sampler.random(n=max_action_changes)

        action_sequences = np.zeros((self.config.T, self.num_actions))

        for i in range(self.num_actions):  # L, then G
            act_low = self.action_space['low'][i]
            act_high = self.action_space['high'][i]
            act_range = act_high - act_low
            
            current_samples = act_low + samples[:, i] * act_range
            
            num_changes = np.random.randint(0, max_action_changes)
            change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
            
            sample_idx = 0
            current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05  # Small variation

            for t in range(self.config.T):
                if len(change_points) > 0 and t == change_points[0]:
                    sample_idx = (sample_idx + 1) % len(current_samples)
                    current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05
                    change_points = change_points[1:]
                action_sequences[t, i] = current_action_val
        
        return action_sequences  # Shape (T, num_actions)

    def generate_x0(self) -> np.ndarray:
        """Generate initial state for the simulation."""
        x0 = np.random.uniform(self.observation_space['low'], self.observation_space['high'] * 0.2, 
                              size=self.observation_space['low'].shape)
        return np.clip(x0, self.observation_space['low'], self.observation_space['high'])

    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        setpoints_dict = self.generate_setpoints()
        disturbances_dict, disturbance_space_for_env = self.generate_disturbances()
        
        unnorm_action_sequence = self.generate_lhs_actions()
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence)
        
        x0 = self.generate_x0()

        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'SP': {},  # No direct setpoint tracking in state vector
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0,
            'model': 'multistage_extraction_reactive',  # For make_env to pick the right model
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'disturbance_bounds': disturbance_space_for_env,
            'disturbances': disturbances_dict,
            'integration_method': 'jax'
        }
        
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        # Store results
        sim_observed_states, sim_disturbance_values, sim_actions = [], [], []
        nl_observed_states, nl_disturbance_values = [], []

        obs_norm, _ = env.reset()
        nl_obs_norm, _ = noiseless_env.reset()
        
        done = False
        step = 0
        while not done:
            action_norm = action_sequence_normalized[step, :]
            
            # Env step returns normalized observations
            obs_norm, _, done, _, _ = env.step(action_norm) 
            nl_obs_norm, _, _, _, _ = noiseless_env.step(action_norm)

            # Split normalized obs into states and disturbances
            current_obs_norm = obs_norm[:self.num_states]
            current_dist_norm = obs_norm[self.num_states : self.num_states + self.num_disturbances]
            
            nl_current_obs_norm = nl_obs_norm[:self.num_states]
            nl_current_dist_norm = nl_obs_norm[self.num_states : self.num_states + self.num_disturbances]

            # Unnormalize
            obs_unnorm = self._unnormalize_observation(current_obs_norm)
            dist_unnorm = self._unnormalize_disturbance(current_dist_norm, disturbance_space_for_env)
            action_unnorm_current_step = self._unnormalize_action(action_norm)
            
            nl_obs_unnorm = self._unnormalize_observation(nl_current_obs_norm)
            nl_dist_unnorm = self._unnormalize_disturbance(nl_current_dist_norm, disturbance_space_for_env)

            sim_observed_states.append(obs_unnorm)
            sim_disturbance_values.append(dist_unnorm)
            sim_actions.append(action_unnorm_current_step)
            
            nl_observed_states.append(nl_obs_unnorm)
            nl_disturbance_values.append(nl_dist_unnorm)
            
            step += 1
            if step >= self.config.T:
                done = True

        # Post-process actions list
        actions_array = np.array(sim_actions)

        return (
            self._format_results(sim_observed_states, sim_disturbance_values, actions_array),
            self._format_results(nl_observed_states, nl_disturbance_values, actions_array)
        )

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        noisy_results, noiseless_results = [], []
        for _ in tqdm(range(self.config.n_simulations), desc="Running Multistage Extraction Reactive Sims"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        num_sims = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sims))

        # Group related states for better visualization
        state_groups = [
            ['XA1', 'XA2', 'XA3', 'XA4', 'XA5'],  # All XA states
            ['YA1', 'YA2', 'YA3', 'YA4', 'YA5'],  # All YA states
            ['YB1', 'YB2', 'YB3', 'YB4', 'YB5'],  # All YB states
            ['YC1', 'YC2', 'YC3', 'YC4', 'YC5']   # All YC states
        ]
        
        action_names = ['L', 'G']
        disturbance_names = ['XA0', 'YA6', 'YB6', 'YC6']

        # Plot each group of states
        for group_name, state_group in zip(['XA States', 'YA States', 'YB States', 'YC States'], state_groups):
            fig, axs = plt.subplots(len(state_group), 1, figsize=(12, 3*len(state_group)), sharex=True)
            plt.suptitle(group_name)
            
            for sim_idx in range(num_sims):
                noisy_res = noisy_results[sim_idx]
                noiseless_res = noiseless_results[sim_idx]
                color = colors[sim_idx]
                label_suffix = f' Sim {sim_idx+1}' if num_sims > 1 else ""
                
                for i, state_name in enumerate(state_group):
                    ax = axs[i] if len(state_group) > 1 else axs
                    ax.plot(noisy_res.observed_states[state_name], color=color, alpha=0.7, 
                            label=f'Noisy{label_suffix}')
                    ax.plot(noiseless_res.observed_states[state_name], color=color, linestyle='--', 
                            alpha=0.7, label=f'Noiseless{label_suffix}')
                    ax.set_title(f'{state_name}')
                    ax.set_ylabel('Concentration')
                    
                    if i == len(state_group) - 1:
                        ax.set_xlabel('Time Step')
                        
            if num_sims > 1:
                axs[0].legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

        # Plot actions
        fig_actions, axs_actions = plt.subplots(len(action_names), 1, figsize=(12, len(action_names) * 3), squeeze=False)
        for i in range(num_sims):
            noisy_res = noisy_results[i]
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""
            for j, act_name in enumerate(action_names):
                axs_actions[j,0].plot(noisy_res.actions[act_name], color=color, label=f'{act_name}{label_suffix}')
                axs_actions[j,0].set_title(f'Action: {act_name}')
                axs_actions[j,0].set_xlabel('Time')
                axs_actions[j,0].set_ylabel('Flow Rate')
        if num_sims > 1:
            for j in range(len(action_names)): 
                axs_actions[j,0].legend(loc='best', fontsize='small')
        fig_actions.tight_layout()
        plt.show()

        # Plot disturbances
        fig_dist, axs_dist = plt.subplots(len(disturbance_names), 1, figsize=(12, len(disturbance_names) * 3), squeeze=False)
        for i in range(num_sims):
            noisy_res = noisy_results[i]
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""
            for j, dist_name in enumerate(disturbance_names):
                axs_dist[j,0].plot(noisy_res.disturbances[dist_name], color=color, label=f'{dist_name}{label_suffix}')
                axs_dist[j,0].set_title(f'Disturbance: {dist_name}')
                axs_dist[j,0].set_xlabel('Time')
                axs_dist[j,0].set_ylabel('Value')
        if num_sims > 1:
            for j in range(len(disturbance_names)): 
                axs_dist[j,0].legend(loc='best', fontsize='small')
        fig_dist.tight_layout()
        plt.show()

    def _normalize_action(self, action_values: np.ndarray) -> np.ndarray:
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

    def _unnormalize_disturbance(self, norm_dist: np.ndarray, disturbance_space: Dict) -> np.ndarray:
        low = disturbance_space['low']
        high = disturbance_space['high']
        return (norm_dist + 1) * (high - low) / 2 + low

    def _format_results(
        self,
        observed_states_list: List[np.ndarray],
        disturbances_list: List[np.ndarray],
        actions_array: np.ndarray
    ) -> SimulationResult:
        
        # Define state names for the reactive model
        state_names = [
            "XA1", "YA1", "YB1", "YC1", 
            "XA2", "YA2", "YB2", "YC2", 
            "XA3", "YA3", "YB3", "YC3", 
            "XA4", "YA4", "YB4", "YC4", 
            "XA5", "YA5", "YB5", "YC5"
        ]
        
        disturbance_names = ["XA0", "YA6", "YB6", "YC6"]
        action_names = ["L", "G"]
        
        # Extract states from observed_states_list
        obs_states_dict = {}
        for i, name in enumerate(state_names):
            obs_states_dict[name] = [state[i] for state in observed_states_list]
        
        # Extract disturbances from disturbances_list
        dist_states_dict = {}
        for i, name in enumerate(disturbance_names):
            dist_states_dict[name] = [dist[i] for dist in disturbances_list]
        
        # Extract actions from actions_array
        action_states_dict = {}
        for i, name in enumerate(action_names):
            action_states_dict[name] = [action[i] for action in actions_array]
        
        return SimulationResult(obs_states_dict, dist_states_dict, action_states_dict)


# Example usage
if __name__ == "__main__":
    config = SimulationConfig(n_simulations=10, T=100, tsim=100, noise_percentage=0.01)
    model_params = {'Vl': 5.0, 'Vg': 5.0, 'm': 1.0, 'Kla': 0.01, 'k': 0.1, 'eq_exponent': 2.0}
    simulator = MultistageExtractionReactiveSimulator(config, model_params=model_params)
    
    noisy_results, noiseless_results = simulator.run_multiple_simulations()
    simulator.plot_results(noisy_results, noiseless_results)
    
    # If you want to convert the simulation results to features and targets:
    converter = MultistageExtractionReactiveConverter()
    features, targets = converter.convert(noisy_results)
    print("Reactive Extraction Features Shape:", features.shape)
    print("Reactive Extraction Targets Shape:", targets.shape)
    
    # Optional: Save to CSV
    # feature_columns = ["XA1", "YA1", "YB1", "YC1", "XA2", "YA2", "YB2", "YC2", 
    #                   "XA3", "YA3", "YB3", "YC3", "XA4", "YA4", "YB4", "YC4", 
    #                   "XA5", "YA5", "YB5", "YC5", "XA0", "YA6", "YB6", "YC6", 
    #                   "L", "G", "simulation_no"]
    # target_columns = ["XA1", "YA1", "YB1", "YC1", "XA2", "YA2", "YB2", "YC2", 
    #                  "XA3", "YA3", "YB3", "YC3", "XA4", "YA4", "YB4", "YC4", 
    #                  "XA5", "YA5", "YB5", "YC5"]
    # 
    # features_df = pd.DataFrame(features, columns=feature_columns)
    # targets_df = pd.DataFrame(targets, columns=target_columns)
    # features_df.to_csv('reactive_extraction_features.csv', index=False)
    # targets_df.to_csv('reactive_extraction_targets.csv', index=False)