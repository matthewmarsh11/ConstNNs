import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy.stats import qmc
import pandas as pd
from collections import defaultdict
from abc import ABC, abstractmethod

# Assuming pcgym is installed and make_env can find the model by string name
from pcgym import make_env
# from pcgym import BaseModel # Or the correct path to BaseModel

# --- Generic Simulation Helper Classes (can be in a shared utils file) ---
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
    tsim: float  # Simulation time period for one run
    noise_percentage: Union[float, Dict[str, float]] = 0.01

class SimulationConverter(ABC):
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert output of simulation and return features and targets"""
        pass

# --- Mass-Spring System Specific Classes ---
class MassSpringConverter(SimulationConverter):
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the mass-spring system simulation data into features and targets."""
        obs_states_list = [res.observed_states for res in data]
        actions_list = [res.actions for res in data]
        # Disturbances are empty for this model

        combined_obs_states = defaultdict(list)
        for obs_sim in obs_states_list:
            for key, values in obs_sim.items():
                combined_obs_states[key].append(values)
        
        combined_actions = defaultdict(list)
        for act_sim in actions_list:
            for key, values in act_sim.items():
                combined_actions[key].append(values)

        for key in combined_obs_states:
            combined_obs_states[key] = np.concatenate(combined_obs_states[key], axis=0)
        for key in combined_actions:
            combined_actions[key] = np.concatenate(combined_actions[key], axis=0)

        feature_keys_obs = ['x1', 'v1', 'x2', 'v2', 'x1_s']
        feature_keys_act = ['F1', 'F2']
        target_keys = ['x1', 'v1', 'x2', 'v2']
        
        if 'x1_s' not in combined_obs_states and 'x1' in combined_obs_states:
            print("Warning: 'x1_s' not found in observed_states. Using 'x1' values as a placeholder.")
            combined_obs_states['x1_s'] = combined_obs_states['x1']

        num_total_timesteps = len(combined_obs_states[feature_keys_obs[0]])

        features_list = []
        for key in feature_keys_obs:
            features_list.append(combined_obs_states[key].reshape(num_total_timesteps, 1))
        for key in feature_keys_act:
            features_list.append(combined_actions[key].reshape(num_total_timesteps, 1))
        
        features = np.hstack(features_list)
        
        targets_list = []
        for key in target_keys:
            targets_list.append(combined_obs_states[key].reshape(num_total_timesteps, 1))
        targets = np.hstack(targets_list)
        
        return features, targets

class MassSpringSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model_name = 'constrained_mass_spring' # pcgym model name

        # Inputs: F1, F2 (External forces)
        self.action_space = {
            'low': np.array([-20.0, -20.0]), # Min force [N]
            'high': np.array([20.0, 20.0])   # Max force [N]
        }
        
        # States: x1, v1, x2, v2. Setpoint: x1_s
        # Observation: [x1, v1, x2, v2, x1_s]
        self.observation_space = {
            'low': np.array([-5.0, -10.0, -5.0, -10.0]), # x_min, v_min, x2_min, v2_min, x1_s_min
            'high': np.array([5.0, 10.0, 5.0, 10.0])    # x_max, v_max, x2_max, v2_max, x1_s_max
            # Units: x [m], v [m/s]
        }
        # Model parameters (defaults from model)
        self.m1 = 1.0; self.m2 = 1.0 
        self.k = 10.0; self.c = 0.5

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for x1."""
        num_changes = np.random.randint(0, self.config.T // 4 + 1)
        change_points = np.sort(np.random.choice(range(1, self.config.T), num_changes, replace=False))
        
        setpoints_x1 = []
        sp_low = self.observation_space['low'][4]  # x1_s lower bound
        sp_high = self.observation_space['high'][4] # x1_s upper bound
        current_setpoint_x1 = np.random.uniform(sp_low, sp_high)
        
        for t_step in range(self.config.T):
            if len(change_points) > 0 and t_step == change_points[0]:
                current_setpoint_x1 = np.random.uniform(sp_low, sp_high)
                change_points = change_points[1:]
            setpoints_x1.append(current_setpoint_x1)
        
        return {'x1': setpoints_x1} # Setpoint for x1

    def generate_disturbances(self) -> Tuple[Dict[str, List[float]], Optional[Dict[str, np.ndarray]]]:
        """No disturbances defined for this model."""
        return {}, None

    def generate_lhs_actions(self) -> np.ndarray:
        """Generate action sequence for F1, F2 using LHS with step changes."""
        sampler = qmc.LatinHypercube(d=len(self.action_space['low'])) # d=2 for F1, F2
        
        max_action_changes = self.config.T // 4 + 1
        samples = sampler.random(n=max_action_changes)
        scaled_samples = self.action_space['low'] + samples * (self.action_space['high'] - self.action_space['low'])
        
        actions_F1, actions_F2 = [], []
        num_changes = np.random.randint(0, max_action_changes)
        change_points = np.sort(np.random.choice(range(1, self.config.T), num_changes, replace=False))
        
        sample_idx = 0
        current_F1 = scaled_samples[sample_idx, 0]
        current_F2 = scaled_samples[sample_idx, 1]
        
        for t_step in range(self.config.T):
            if len(change_points) > 0 and t_step == change_points[0]:
                sample_idx = (sample_idx + 1) % max_action_changes
                current_F1 = scaled_samples[sample_idx, 0]
                current_F2 = scaled_samples[sample_idx, 1]
                change_points = change_points[1:]
            actions_F1.append(current_F1)
            actions_F2.append(current_F2)
            
        return np.array([actions_F1, actions_F2]).T # Shape (T, 2)

    def generate_x0(self) -> np.ndarray:
        """Generate initial state [x1, v1, x2, v2] satisfying x1+x2=0, v1+v2=0."""
        x1_init = np.random.uniform(self.observation_space['low'][0] * 0.5, self.observation_space['high'][0] * 0.5) # Smaller range for init
        v1_init = np.random.uniform(self.observation_space['low'][1] * 0.5, self.observation_space['high'][1] * 0.5)
        
        x2_init = -x1_init
        v2_init = -v1_init
        
        return np.array([x1_init, v1_init, x2_init, v2_init])

    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        """Run a single simulation."""
        unnorm_action_sequence = self.generate_lhs_actions()
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence)
        
        x0_physical = self.generate_x0() # Physical states [x1, v1, x2, v2]
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'SP': {},
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0_physical,
            'model': self.model_name,
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'custom_model': None,
        }
        
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        obs_list_noisy, actions_list_noisy = [], []
        obs_list_noiseless, actions_list_noiseless = [], []

        obs_norm, _ = env.reset()
        noiseless_obs_norm, _ = noiseless_env.reset()
        
        for step_num in range(self.config.T):
            action_normalized = action_sequence_normalized[step_num, :]
            
            obs_norm, _, done, _, info = env.step(action_normalized)
            noiseless_obs_norm, _, _, _, _ = noiseless_env.step(action_normalized)
            
            obs_unnorm = self._unnormalize_observation(obs_norm)
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs_norm)
            action_unnorm = unnorm_action_sequence[step_num, :]
            
            obs_list_noisy.append(obs_unnorm)
            actions_list_noisy.append(action_unnorm)

            obs_list_noiseless.append(noiseless_obs_unnorm)
            actions_list_noiseless.append(action_unnorm)

            if done:
                break
        
        empty_disturbances_values = {}
        return (
            self._format_results(obs_list_noisy, actions_list_noisy),
            self._format_results(obs_list_noiseless, actions_list_noiseless)
        )

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        noisy_results, noiseless_results = [], []
        for _ in tqdm(range(self.config.n_simulations), desc=f"Running {self.model_name} simulations"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]):
        if not noisy_results: return
        num_simulations = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_simulations))

        state_keys = ['x1', 'v1', 'x2', 'v2', 'x1_s']
        state_titles = ['Position x1 (m)', 'Velocity v1 (m/s)', 
                        'Position x2 (m)', 'Velocity v2 (m/s)', 
                        'Setpoint x1_s (m)']
        
        fig_states, axs_states = plt.subplots(3, 2, figsize=(15, 12), squeeze=False)
        axs_states_flat = axs_states.flatten()

        for i in range(num_simulations):
            color, noisy_sim, noiseless_sim = colors[i], noisy_results[i], noiseless_results[i]
            label_suffix = f"Sim {i+1}"
            for j, key in enumerate(state_keys):
                if key in noisy_sim.observed_states and key in noiseless_sim.observed_states:
                    axs_states_flat[j].plot(noisy_sim.observed_states[key], color=color, alpha=0.8, label=f"Noisy {label_suffix}")
                    axs_states_flat[j].plot(noiseless_sim.observed_states[key], color=color, linestyle='--', alpha=0.8, label=f"Noiseless {label_suffix}")
        
        for j, title in enumerate(state_titles):
            axs_states_flat[j].set_title(title)
            axs_states_flat[j].set_xlabel("Time step")
            axs_states_flat[j].legend(fontsize='small', loc='best')

        if len(state_keys) < len(axs_states_flat):
            for k in range(len(state_keys), len(axs_states_flat)):
                fig_states.delaxes(axs_states_flat[k])

        fig_states.tight_layout()
        plt.show(block=False)

        action_keys = ['F1', 'F2']
        action_titles = ['Force F1 (N)', 'Force F2 (N)']
        fig_actions, axs_actions = plt.subplots(len(action_keys), 1, figsize=(10, 4 * len(action_keys)), squeeze=False)
        
        for i in range(num_simulations):
            color, noisy_sim = colors[i], noisy_results[i]
            label_suffix = f"Sim {i+1}"
            for j, key in enumerate(action_keys):
                if key in noisy_sim.actions:
                    axs_actions[j, 0].plot(noisy_sim.actions[key], color=color, alpha=0.8, label=f"{label_suffix}")

        for j, title in enumerate(action_titles):
            axs_actions[j, 0].set_title(title)
            axs_actions[j, 0].set_xlabel("Time step")
            axs_actions[j, 0].legend(fontsize='small', loc='best')
        
        fig_actions.tight_layout()
        plt.show(block=False)
        
    def plot_constraint_residuals(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]):
        """Plots the residual of the linear constraint x1 + x2 = 0."""
        if not noisy_results:
            print("No simulation results for mass-spring constraint residuals.")
            return

        num_simulations = len(noisy_results)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_simulations))
        fig_residuals, ax_residual = plt.subplots(1, 1, figsize=(12, 6))
        
        for i in range(num_simulations):
            color = colors[i]
            noisy_sim = noisy_results[i]
            if 'x1' in noisy_sim.observed_states and 'x2' in noisy_sim.observed_states:
                x1_n = np.array(noisy_sim.observed_states['x1'])
                x2_n = np.array(noisy_sim.observed_states['x2'])
                if len(x1_n) == len(x2_n) and len(x1_n) > 0:
                    residual_n = x1_n + x2_n
                    ax_residual.plot(residual_n, color=color, linestyle='-', alpha=0.7, label=f"Noisy Sim {i+1}" if i == 0 else None)

            noiseless_sim = noiseless_results[i]
            if 'x1' in noiseless_sim.observed_states and 'x2' in noiseless_sim.observed_states:
                x1_nl = np.array(noiseless_sim.observed_states['x1'])
                x2_nl = np.array(noiseless_sim.observed_states['x2'])
                if len(x1_nl) == len(x2_nl) and len(x1_nl) > 0:
                    residual_nl = x1_nl + x2_nl
                    ax_residual.plot(residual_nl, color=color, linestyle='--', alpha=0.7, label=f"Noiseless Sim {i+1}" if i == 0 else None)
        
        ax_residual.set_title('Mass-Spring System Constraint Residual ($x_1 + x_2$)')
        ax_residual.set_xlabel('Time step')
        ax_residual.set_ylabel('Residual (m)')
        ax_residual.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax_residual.legend(fontsize='small', loc='best')
        fig_residuals.tight_layout()
        plt.show(block=False)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        return 2 * (action - self.action_space['low']) / (self.action_space['high'] - self.action_space['low']) - 1

    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return (action_norm + 1) * (self.action_space['high'] - self.action_space['low']) / 2 + self.action_space['low']

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        return 2 * (obs - self.observation_space['low']) / (self.observation_space['high'] - self.observation_space['low']) - 1
    
    def _unnormalize_observation(self, obs_norm: np.ndarray) -> np.ndarray:
        return (obs_norm + 1) * (self.observation_space['high'] - self.observation_space['low']) / 2 + self.observation_space['low']

    def _format_results(
        self,
        observed_values_list: List[np.ndarray],
        action_values_list: List[np.ndarray]
    ) -> SimulationResult:
        obs_states_dict = defaultdict(list)
        if observed_values_list:
            obs_keys = ['x1', 'v1', 'x2', 'v2', 'x1_s']
            for step_obs_array in observed_values_list:
                for i, key in enumerate(obs_keys):
                    if i < step_obs_array.shape[0]:
                         obs_states_dict[key].append(step_obs_array[i])
        
        action_states_dict = defaultdict(list)
        if action_values_list:
            act_keys = ['F1', 'F2']
            for step_act_array in action_values_list:
                for i, key in enumerate(act_keys):
                    if i < step_act_array.shape[0]:
                        action_states_dict[key].append(step_act_array[i])
        
        return SimulationResult(dict(obs_states_dict), dict(action_states_dict))

# --- Main script execution ---
if __name__ == "__main__":
    np.random.seed(123) # Different seed

    config = SimulationConfig(
        n_simulations=2,
        T=300,            # Longer simulation for spring dynamics
        tsim=30.0,        # Simulation time (e.g., seconds)
        noise_percentage=0.005 # Smaller noise for mechanical system
    )

    simulator = MassSpringSimulator(config)
    noisy_sim_results, noiseless_sim_results = simulator.run_multiple_simulations()
    simulator.plot_results(noisy_sim_results, noiseless_sim_results)
    simulator.plot_constraint_residuals(noisy_sim_results, noiseless_sim_results) # New call
    plt.show() 

    converter = MassSpringConverter()
    features, targets = converter.convert(noisy_sim_results)
    noiseless_features, _ = converter.convert(noiseless_sim_results) # Usually targets are from noisy

    features_df = pd.DataFrame(features, columns=['x1', 'v1', 'x2', 'v2', 'x1_s', 'F1', 'F2'])
    targets_df = pd.DataFrame(targets, columns=['x1', 'v1', 'x2', 'v2'])
    noiseless_df = pd.DataFrame(noiseless_features, columns=['x1', 'v1', 'x2', 'v2', 'x1_s', 'F1', 'F2'])

    # features_df.to_csv('mass_spring_features.csv', index=False)
    # targets_df.to_csv('mass_spring_targets.csv', index=False)
    # noiseless_df.to_csv('mass_spring_noiseless_features.csv', index=False)
    
    # print("Mass-spring system simulation data saved to CSV files.")