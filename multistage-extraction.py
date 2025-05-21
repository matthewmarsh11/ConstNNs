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
    


class MultistageExtractionConverter(SimulationConverter):
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
        # These are the names from _format_results
        feature_keys_obs = ["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "X5", "Y5"]
        feature_keys_dist = ["X0_feed", "Y6_feed"]
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
        timesteps_per_simulation = len(data[0].observed_states["X1"]) # Assuming all states have same length
        
        # Simulation numbers: shape (num_simulations, timesteps_per_simulation)
        sim_no_array = np.array([[i + 1] * timesteps_per_simulation for i in range(num_simulations)])
        all_features_dict['simulation_no'] = sim_no_array

        # Stack features: result shape (num_simulations, timesteps, num_features)
        # Order of features matters for the final array
        ordered_feature_keys = feature_keys_obs + feature_keys_dist + feature_keys_act + ['simulation_no']
        features_stacked = np.stack([all_features_dict[key] for key in ordered_feature_keys], axis=-1)
        
        # Reshape to (total_samples, num_features)
        features = features_stacked.reshape(-1, features_stacked.shape[-1])

        # Define targets: e.g., all state variables at the next time step or specific outputs
        # For now, let's assume targets are the observed states themselves (for autoencoder, or one-step prediction)
        target_keys = ["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "X5", "Y5"] # Example targets
        
        targets_stacked = np.stack([combined_obs_states[key] for key in target_keys], axis=-1)
        targets = targets_stacked.reshape(-1, targets_stacked.shape[-1])
        
        return features, targets

class MultistageExtractionSimulator:
    def __init__(self, config: SimulationConfig, model_params: Optional[Dict] = None):
        self.config = config

        # Define spaces (adjust ranges as needed)
        # Inputs: L (Liquid flowrate), G (Gas flowrate)
        self.action_space = {
            'low': np.array([1, 1]),    # L_min, G_min
            'high': np.array([5, 5])  # L_max, G_max
        }
        # States: X1..X5, Y1..Y5 (Concentrations, typically 0-1 or within physical bounds)
        self.observation_space = {
            'low': np.array([0.0] * 10),
            'high': np.array([1.0] * 10)
        }
        # Disturbances: X0_feed (Feed liquid conc), Y6_feed (Feed gas conc)
        self.disturbance_definition = { # For use in generate_disturbances
            'X0': {'low': 0.3, 'high': 0.9, 'idx': 0},
            'Y6': {'low': 0.01, 'high': 0.2, 'idx': 1}
        }
        # For internal _unnormalize_disturbance and env_params['disturbance_bounds']
        self.disturbance_space_env = {
            'low': np.array([self.disturbance_definition['X0']['low'], self.disturbance_definition['Y6']['low']]),
            'high': np.array([self.disturbance_definition['X0']['high'], self.disturbance_definition['Y6']['high']])
        }

        self.num_actions = len(self.action_space['low'])
        self.num_states = len(self.observation_space['low'])
        self.num_disturbances = len(self.disturbance_definition)

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for a target state, e.g., X5."""
        # This is optional; for extraction, setpoints might not be as common as for CSTR temperature.
        # If used, one might target X5 or Y1.
        # For now, let's generate a dummy setpoint for X5_s to match CSTR structure if needed,
        # but it won't be actively used by the simple extraction model control here.
        num_changes = np.random.randint(0, self.config.T // 10) # Fewer changes for setpoints
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
        
        setpoints_X5 = []
        # Assuming X5 is a concentration, typically between 0 and 1.
        # Use a narrower range for setpoints than the full observation space.
        sp_low, sp_high = self.observation_space['low'][8]*1.1, self.observation_space['high'][8]*0.9 # for X5
        current_setpoint = np.random.uniform(sp_low, sp_high) if sp_low < sp_high else (sp_low+sp_high)/2
        
        for t in range(self.config.T):
            if len(change_points) > 0 and t == change_points[0]:
                current_setpoint = np.random.uniform(sp_low, sp_high) if sp_low < sp_high else (sp_low+sp_high)/2
                change_points = change_points[1:]
            setpoints_X5.append(current_setpoint)
        
        # The environment might expect setpoints for all states or specific ones.
        # The CSTR example had 'Ca_s'. If pcgym requires setpoints for all observable states,
        # this needs to be expanded. For now, just one example.
        # The current pcgym dummy env doesn't strictly use it beyond x0 if Ca_s is the 5th element of x0.
        # Let's return it in a way it can be part of x0 if needed by pcgym.
        # The CSTR simulator adds Ca_s to obs, then the converter removes it.
        # To match, we can have an X5_s.
        return {'X5_s': setpoints_X5}


    def generate_disturbances(self) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
        """Generate random disturbances for X0_feed, Y6_feed using LHS."""
        disturbance_space = {'low': np.array([0.3, 0.01]), 'high': np.array([0.9, 0.2])} # X0_feed, Y6_feed
        
        sampler = qmc.LatinHypercube(d=len(disturbance_space['low'])) # For X0_feed, Y6_feed
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
            current_val = d_samples[sample_idx] + (np.random.rand() - 0.5) * d_range * 0.05 # Small variation

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
 # 2 for L and G
        sampler = qmc.LatinHypercube(d=self.num_actions) # For L, G
        max_action_changes = self.config.T // 4
        samples = sampler.random(n=max_action_changes)

        action_sequences = np.zeros((self.config.T, self.num_actions))

        for i in range(self.num_actions): # L, then G
            act_low = self.action_space['low'][i]
            act_high = self.action_space['high'][i]
            act_range = act_high - act_low
            
            current_samples = act_low + samples[:, i] * act_range
            
            num_changes = np.random.randint(0, max_action_changes)
            change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
            
            sample_idx = 0
            current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05 # Small variation

            for t in range(self.config.T):
                if len(change_points) > 0 and t == change_points[0]:
                    sample_idx = (sample_idx + 1) % len(current_samples)
                    current_action_val = current_samples[sample_idx] + (np.random.rand() - 0.5) * act_range * 0.05
                    change_points = change_points[1:]
                action_sequences[t, i] = current_action_val
        
        return action_sequences # Shape (T, num_actions)

    def generate_x0(self) -> np.ndarray:
        """Generate initial state for the simulation."""
        # For concentrations, a slight random deviation from a low or mid value.
        # midpoints = (self.observation_space['high'] + self.observation_space['low']) / 2
        # deviations = (np.random.rand(*self.observation_space['low'].shape) - 0.5) * \
        #              (self.observation_space['high'] - self.observation_space['low']) * 0.1
        # x0 = midpoints + deviations
        # A simple start:
        x0 = np.random.uniform(self.observation_space['low'], self.observation_space['high'] * 0.2, size=self.observation_space['low'].shape)
        return np.clip(x0, self.observation_space['low'], self.observation_space['high'])

    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        setpoints_dict = self.generate_setpoints() # e.g. {'X5_s': [...]}
        disturbances_dict, disturbance_space_for_env = self.generate_disturbances()
        
        unnorm_action_sequence = self.generate_lhs_actions() # (T, num_actions)
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence) # (T, num_actions)
        
        x0 = self.generate_x0() # (num_states,)

        # The CSTR env_params SP was a dict {'Ca': setpoints_list}.
        # If setpoints are to be tracked as separate states (like Ca_s), they need to be part of x0
        # and handled by the observation space. For this model, let's assume setpoints are for external reference
        # or a controller not yet implemented, and not part of the core state vector passed to the model.
        # The 'SP' in env_params for pcgym is often for the controller within the env.
        # For data generation, we mainly care about x0 for the process states.
        # If 'X5_s' is to be treated like 'Ca_s', the observation space and state vector need to include it.
        # The CSTR example observation space is [Ca, Cb, Cc, T, Ca_s].
        # Let's simplify and not add X5_s to the state vector for this extraction model,
        # as it's not in the multistage_extraction model's state definition.
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'SP': {}, # For now, no direct setpoint tracking in state vector like CSTR
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0, # Initial states for X1...Y5
            'model': 'multistage_extraction', # For make_env to potentially pick the right model class if registered
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'disturbance_bounds': disturbance_space_for_env,
            'disturbances': disturbances_dict, # Time series for each disturbance
        }
        
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        # Store results
        sim_observed_states, sim_disturbance_values, sim_actions = [], [], []
        nl_observed_states, nl_disturbance_values = [], []

        obs_norm, _ = env.reset() # obs_norm includes states + disturbances (normalized)
        nl_obs_norm, _ = noiseless_env.reset()
        
        done = False
        step = 0
        while not done:
            action_norm = action_sequence_normalized[step, :] # Get current normalized action vector
            
            # Env step returns normalized observations
            obs_norm, _, done, _, _ = env.step(action_norm) 
            nl_obs_norm, _, _, _, _ = noiseless_env.step(action_norm)

            # Split normalized obs into states and disturbances
            # pcgym returns [states_norm, disturbances_norm, uncertainties_norm (if any)]
            current_obs_norm = obs_norm[:self.num_states]
            current_dist_norm = obs_norm[self.num_states : self.num_states + self.num_disturbances]
            
            nl_current_obs_norm = nl_obs_norm[:self.num_states]
            nl_current_dist_norm = nl_obs_norm[self.num_states : self.num_states + self.num_disturbances]

            # Unnormalize
            obs_unnorm = self._unnormalize_observation(current_obs_norm)
            dist_unnorm = self._unnormalize_disturbance(current_dist_norm, disturbance_space_for_env)
            action_unnorm_current_step = self._unnormalize_action(action_norm) # This is a single step action vector
            
            nl_obs_unnorm = self._unnormalize_observation(nl_current_obs_norm)
            nl_dist_unnorm = self._unnormalize_disturbance(nl_current_dist_norm, disturbance_space_for_env)

            sim_observed_states.append(obs_unnorm)
            sim_disturbance_values.append(dist_unnorm)
            sim_actions.append(action_unnorm_current_step) # Store the single action vector for this step
            
            nl_observed_states.append(nl_obs_unnorm)
            nl_disturbance_values.append(nl_dist_unnorm)
            
            step += 1
            if step >= self.config.T: # Ensure termination
                done = True

        # Post-process actions list: it's a list of np.array([val_L, val_G]), needs to be (T, num_actions) then formatted
        actions_array = np.array(sim_actions) # Should be (T, num_actions)

        return (
            self._format_results(sim_observed_states, sim_disturbance_values, actions_array),
            self._format_results(nl_observed_states, nl_disturbance_values, actions_array) # Same actions for both
        )

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        noisy_results, noiseless_results = [], []
        for _ in tqdm(range(self.config.n_simulations), desc="Running Multistage Extraction Sims"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        num_sims = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sims))

        state_names = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5']
        action_names = ['L', 'G']
        disturbance_names = ['X0', 'Y6']

        num_state_plots = len(state_names)
        fig_states, axs_states = plt.subplots((num_state_plots + 1) // 2, 2, figsize=(15, num_state_plots * 2), squeeze=False)
        axs_states_flat = axs_states.flatten()

        for i in range(num_sims):
            noisy_res = noisy_results[i]
            noiseless_res = noiseless_results[i]
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""

            for j, state_name in enumerate(state_names):
                axs_states_flat[j].plot(noisy_res.observed_states[state_name], color=color, alpha=0.8, label=f'Noisy{label_suffix}')
                axs_states_flat[j].plot(noiseless_res.observed_states[state_name], color=color, linestyle='--', alpha=0.8, label=f'Noiseless{label_suffix}')
                axs_states_flat[j].set_title(state_name)
                axs_states_flat[j].set_xlabel('Time')
                axs_states_flat[j].set_ylabel('Concentration')

        for j in range(num_state_plots):
             if num_sims > 1 : axs_states_flat[j].legend(loc='best', fontsize='small')
        fig_states.tight_layout()
        plt.show()

        fig_actions, axs_actions = plt.subplots(len(action_names), 1, figsize=(12, len(action_names) * 3), squeeze=False)
        for i in range(num_sims):
            noisy_res = noisy_results[i] # Actions are same for noisy/noiseless typically
            color = colors[i]
            label_suffix = f' Sim {i+1}' if num_sims > 1 else ""
            for j, act_name in enumerate(action_names):
                axs_actions[j,0].plot(noisy_res.actions[act_name], color=color, label=f'{act_name}{label_suffix}')
                axs_actions[j,0].set_title(f'Action: {act_name}')
                axs_actions[j,0].set_xlabel('Time')
                axs_actions[j,0].set_ylabel('Flow Rate')
        if num_sims > 1:
            for j in range(len(action_names)): axs_actions[j,0].legend(loc='best', fontsize='small')
        fig_actions.tight_layout()
        plt.show()

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
            for j in range(len(disturbance_names)): axs_dist[j,0].legend(loc='best', fontsize='small')
        fig_dist.tight_layout()
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

    def _unnormalize_disturbance(self, norm_dist: np.ndarray, disturbance_space: Dict) -> np.ndarray:
        low = disturbance_space['low'] # Using the specific disturbance_space passed
        high = disturbance_space['high']
        return (norm_dist + 1) * (high - low) / 2 + low

    def _format_results(
        self,
        observed_states_list: List[np.ndarray], # List of state vectors [X1,..Y5] per time step
        disturbances_list: List[np.ndarray],    # List of disturbance vectors [X0f, Y6f] per time step
        actions_array: np.ndarray              # Array of action vectors (T, num_actions) [L, G]
    ) -> SimulationResult:
        
        num_timesteps = len(observed_states_list)
        
        obs_states_dict = {'X1': [state[0] for state in observed_states_list],
                            'Y1': [state[1] for state in observed_states_list],
                            'X2': [state[2] for state in observed_states_list],
                            'Y2': [state[3] for state in observed_states_list],
                            'X3': [state[4] for state in observed_states_list],
                            'Y3': [state[5] for state in observed_states_list],
                            'X4': [state[6] for state in observed_states_list],
                            'Y4': [state[7] for state in observed_states_list],
                            'X5': [state[8] for state in observed_states_list],
                            'Y5': [state[9] for state in observed_states_list]}
        
        dist_states_dict = {'X0': [state[0] for state in disturbances_list],
                            'Y6': [state[1] for state in disturbances_list]}
        
        
        # actions_array is already (T, num_actions)
        action_states_dict = {'L': [state[0] for state in actions_array],
                              'G': [state[1] for state in actions_array]}
        
        return SimulationResult(obs_states_dict, dist_states_dict, action_states_dict)

# --- Example Usage for Multistage Extraction ---
config_ext = SimulationConfig(n_simulations=10, T=100, tsim=100, noise_percentage=0.01)
model_parameters_ext = {'Vl': 6, 'Vg': 4, 'Kla': 5.5} # Example of passing model params
simulator_ext = MultistageExtractionSimulator(config_ext, model_params=model_parameters_ext)

noisy_results_ext, noiseless_results_ext = simulator_ext.run_multiple_simulations()
simulator_ext.plot_results(noisy_results_ext, noiseless_results_ext)

# converter_ext = MultistageExtractionConverter()
# features_ext, targets_ext = converter_ext.convert(noisy_results_ext)
# # To get noiseless features/targets if needed:
# # noiseless_features_ext, noiseless_targets_ext = converter_ext.convert(noiseless_results_ext) 

# print("Multistage Extraction Features Shape:", features_ext.shape)
# print("Multistage Extraction Targets Shape:", targets_ext.shape)

# # Save to CSV (optional, update paths)
# features_df_ext = pd.DataFrame(features_ext, columns=simulator_ext.model_instance.states + simulator_ext.model_instance.disturbances + simulator_ext.model_instance.inputs + ['simulation_no'])
# target_names_ext = simulator_ext.model_instance.states # Assuming targets are the states
# targets_df_ext = pd.DataFrame(targets_ext, columns=target_names_ext)

# # features_df_ext.to_csv('multistage_extraction_features.csv', index=False)
# # targets_df_ext.to_csv('multistage_extraction_targets.csv', index=False)