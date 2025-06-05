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
# If BaseModel is needed for type hinting and pcgym.BaseModel is the path:
# from pcgym import BaseModel # Or the correct path to BaseModel

# --- Generic Simulation Helper Classes (can be in a shared utils file) ---
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
    tsim: float  # Simulation time period for one run
    noise_percentage: Union[float, Dict[str, float]] = 0.01
    # Default V_total from the model, can be overridden if needed
    V_total: float = 100.0 

class SimulationConverter(ABC):
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert output of simulation and return features and targets"""
        pass

# --- Tank System Specific Classes ---
class TankSystemConverter(SimulationConverter):
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the tank system simulation data into features and targets."""
        obs_states_list = [res.observed_states for res in data]
        actions_list = [res.actions for res in data]
        # Disturbances are empty for this model as per definition
        
        # Combine data from all simulations
        combined_obs_states = defaultdict(list)
        for obs_sim in obs_states_list:
            for key, values in obs_sim.items():
                combined_obs_states[key].append(values)
        
        combined_actions = defaultdict(list)
        for act_sim in actions_list:
            for key, values in act_sim.items():
                combined_actions[key].append(values)

        # Features: V1, C1, V2, C2, V1_s (setpoint), F_in1, F_in2
        # Targets: V1, C1, V2, C2 (next state, or current state if predicting dynamics)
        # Here, we'll use current states, SP, and actions as features, and current states as targets (like CSTR example)

        # Ensure all values are numpy arrays of lists of trajectories
        # Then concatenate trajectories from different simulations
        for key in combined_obs_states:
            combined_obs_states[key] = np.concatenate(combined_obs_states[key], axis=0)
        for key in combined_actions:
            combined_actions[key] = np.concatenate(combined_actions[key], axis=0)

        feature_keys_obs = ['V1', 'C1', 'V2', 'C2', 'V1_s']
        feature_keys_act = ['F_in1', 'F_in2']
        target_keys = ['V1', 'C1', 'V2', 'C2']

        # Check if V1_s is present, if not, generate a dummy one or error
        if 'V1_s' not in combined_obs_states and 'V1' in combined_obs_states:
            print("Warning: 'V1_s' not found in observed_states. Using 'V1' values as a placeholder for 'V1_s'.")
            combined_obs_states['V1_s'] = combined_obs_states['V1'] # Placeholder

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

class TankSystemSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model_name = 'constrained_tank_system' # pcgym model name

        # Define spaces based on typical ranges for the tank system
        # Inputs: F_in1, F_in2 (Inlet flow rates)
        self.action_space = {
            'low': np.array([0.1, 0.1]),    # Min flow rate [m³/min]
            'high': np.array([20.0, 20.0])  # Max flow rate [m³/min]
        }
        
        # States: V1, C1, V2, C2. Setpoint: V1_s
        # Observation: [V1, C1, V2, C2, V1_s]
        v_epsilon = 1.0 # Min volume to avoid division by zero if V is in denominator
        self.observation_space = {
            'low': np.array([v_epsilon, 0.0, v_epsilon, 0.0]),
            'high': np.array([config.V_total - v_epsilon, 5.0, config.V_total - v_epsilon, 5.0])
            # Max C of 5.0 kg/m³ is an assumption
        }
        # Model parameters (defaults from model, can be fetched if pcgym allows)
        self.V_total = config.V_total 
        self.C_in1 = 2.0  # [kg/m³]
        self.C_in2 = 3.0  # [kg/m³]

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for V1."""
        num_changes = np.random.randint(0, self.config.T // 4 + 1) # Max T//4 changes
        change_points = np.sort(np.random.choice(range(1, self.config.T), num_changes, replace=False)) # T is steps, tsim is time
        
        setpoints_V1 = []
        # Setpoint is within observation space bounds for V1_s
        sp_low = self.observation_space['low'][4]
        sp_high = self.observation_space['high'][4]
        current_setpoint_V1 = np.random.uniform(sp_low, sp_high)
        
        for t_step in range(self.config.T):
            if len(change_points) > 0 and t_step == change_points[0]:
                current_setpoint_V1 = np.random.uniform(sp_low, sp_high)
                change_points = change_points[1:]
            setpoints_V1.append(current_setpoint_V1)
        
        return {'V1': setpoints_V1} # Setpoint for V1



    def generate_lhs_actions(self) -> np.ndarray:
        """Generate action sequence using Latin Hypercube Sampling with step changes for F_in1, F_in2."""
        sampler = qmc.LatinHypercube(d=len(self.action_space['low'])) # d=2 for F_in1, F_in2
        
        max_action_changes = self.config.T // 4 +1
        samples = sampler.random(n=max_action_changes)
        
        scaled_samples = self.action_space['low'] + samples * (self.action_space['high'] - self.action_space['low'])
        
        actions_F_in1 = []
        actions_F_in2 = []
        
        num_changes = np.random.randint(0, max_action_changes)
        change_points = np.sort(np.random.choice(range(1, self.config.T), num_changes, replace=False))
        
        sample_idx = 0
        current_action_F_in1 = scaled_samples[sample_idx, 0]
        current_action_F_in2 = scaled_samples[sample_idx, 1]
        
        for t_step in range(self.config.T):
            if len(change_points) > 0 and t_step == change_points[0]:
                sample_idx = (sample_idx + 1) % max_action_changes
                current_action_F_in1 = scaled_samples[sample_idx, 0]
                current_action_F_in2 = scaled_samples[sample_idx, 1]
                change_points = change_points[1:]
            actions_F_in1.append(current_action_F_in1)
            actions_F_in2.append(current_action_F_in2)
            
        return np.array([actions_F_in1, actions_F_in2]).T # Shape (T, 2)

    def generate_x0(self) -> np.ndarray:
        """Generate initial state [V1, C1, V2, C2] satisfying V1+V2=V_total."""
        # V1 from a smaller range than full [0, V_total] to avoid issues at boundaries
        v1_min_init = self.observation_space['low'][0] 
        v1_max_init = self.observation_space['high'][0]
        
        V1_init = np.random.uniform(v1_min_init, v1_max_init)
        V2_init = self.V_total - V1_init

        # Ensure V2_init is also within reasonable bounds if V1 is picked near extremes
        # This should be implicitly handled if v1_min/max_init are sensible (e.g. not 0 or V_total)
        V2_init = np.clip(V2_init, self.observation_space['low'][2], self.observation_space['high'][2])
        V1_init = self.V_total - V2_init # Re-adjust V1 if V2 was clipped

        C1_init = np.random.uniform(self.observation_space['low'][1], self.observation_space['high'][1])
        C2_init = np.random.uniform(self.observation_space['low'][3], self.observation_space['high'][3])
        
        return np.array([V1_init, C1_init, V2_init, C2_init])

    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        """Run a single simulation."""
        # setpoints = self.generate_setpoints()
        unnorm_action_sequence = self.generate_lhs_actions() # Already in true scale
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence) # Normalize for env
        
        x0_physical = self.generate_x0() # Physical states [V1, C1, V2, C2]
        
        env_params = {
            'N': self.config.T,         # Number of steps
            'tsim': self.config.tsim,   # Total simulation time for these N steps
            'SP': {},
            'o_space': self.observation_space, # For normalization inside env if it uses it
            'a_space': self.action_space,     # For normalization inside env
            'x0': x0_physical,
            'model': self.model_name,
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True, # pcgym normalizes observations based on o_space
            'normalise_a': True, # pcgym unnormalizes actions based on a_space
            # No disturbances defined in model, so these are empty/None
            'custom_model': None, 
        }
        
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        obs_list_noisy, actions_list_noisy = [], []
        obs_list_noiseless, actions_list_noiseless = [], []

        obs_norm, _ = env.reset() # obs_norm is [V1,C1,V2,C2,V1_s] (normalized by pcgym)
        noiseless_obs_norm, _ = noiseless_env.reset()
        
        for step_num in range(self.config.T):
            action_normalized = action_sequence_normalized[step_num, :]
            
            obs_norm, _, done, _, info = env.step(action_normalized)
            noiseless_obs_norm, _, _, _, _ = noiseless_env.step(action_normalized)
            
            # pcgym observation `obs_norm` is [state1, ..., stateN, sp1, ...] (normalized)
            # No disturbances are appended by pcgym if not provided in env_params['disturbances']
            
            # Unnormalize observations (states + setpoint)
            obs_unnorm = self._unnormalize_observation(obs_norm)
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs_norm)
            
            # Actions are already unnormalized (original scale) from unnorm_action_sequence
            action_unnorm = unnorm_action_sequence[step_num, :] 
            
            obs_list_noisy.append(obs_unnorm)
            actions_list_noisy.append(action_unnorm)
            # disturbance_list_noisy.append(...) # No disturbances

            obs_list_noiseless.append(noiseless_obs_unnorm)
            actions_list_noiseless.append(action_unnorm) # Same actions
            # disturbance_list_noiseless.append(...)

            if done:
                break
        
        # Disturbances are empty for this model
        empty_disturbances_values = {key: [] for key in []} # Example if there were dist keys

        return (
            self._format_results(obs_list_noisy, empty_disturbances_values, actions_list_noisy),
            self._format_results(obs_list_noiseless, empty_disturbances_values, actions_list_noiseless)
        )

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        noisy_results, noiseless_results = [], []
        for _ in tqdm(range(self.config.n_simulations), desc=f"Running {self.model_name} simulations"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]):
        if not noisy_results:
            print("No simulation results to plot.")
            return

        num_simulations = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_simulations))

        # States: V1, C1, V2, C2, V1_s (5 states)
        state_keys = ['V1', 'C1', 'V2', 'C2', 'V1_s']
        state_titles = ['Volume V1 ($m^3$)', 'Concentration C1 (kg/$m^3$)', 
                        'Volume V2 ($m^3$)', 'Concentration C2 (kg/$m^3$)', 
                        'Setpoint V1_s ($m^3$)']
        
        fig_states, axs_states = plt.subplots(3, 2, figsize=(15, 12), squeeze=False)
        axs_states_flat = axs_states.flatten()

        for i in range(num_simulations):
            color = colors[i]
            noisy_sim = noisy_results[i]
            noiseless_sim = noiseless_results[i]
            label_suffix = f"Sim {i+1}"

            for j, key in enumerate(state_keys):
                if key in noisy_sim.observed_states and key in noiseless_sim.observed_states:
                    axs_states_flat[j].plot(noisy_sim.observed_states[key], color=color, alpha=0.8, label=f"Noisy {label_suffix}")
                    axs_states_flat[j].plot(noiseless_sim.observed_states[key], color=color, linestyle='--', alpha=0.8, label=f"Noiseless {label_suffix}")
        
        for j, title in enumerate(state_titles):
            axs_states_flat[j].set_title(title)
            axs_states_flat[j].set_xlabel("Time step")
            axs_states_flat[j].legend(fontsize='small', loc='best')
        
        if len(state_keys) < len(axs_states_flat): # Hide unused subplots
            for k in range(len(state_keys), len(axs_states_flat)):
                fig_states.delaxes(axs_states_flat[k])
        
        fig_states.tight_layout()
        plt.show(block=False)

        # Actions: F_in1, F_in2 (2 actions)
        action_keys = ['F_in1', 'F_in2']
        action_titles = ['Inlet Flow F_in1 ($m^3$/min)', 'Inlet Flow F_in2 ($m^3$/min)']
        
        fig_actions, axs_actions = plt.subplots(len(action_keys), 1, figsize=(10, 4 * len(action_keys)), squeeze=False)
        
        for i in range(num_simulations):
            color = colors[i]
            noisy_sim = noisy_results[i] # Actions are same for noisy/noiseless run
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
        
        # No disturbances to plot for this model

    def plot_constraint_residuals(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]):
            """Plots the residual of the linear constraint V1 + V2 - V_total."""
            if not noisy_results:
                print("No simulation results to plot constraint residuals for.")
                return

            num_simulations = len(noisy_results)
            colors = plt.cm.coolwarm(np.linspace(0, 1, num_simulations))

            fig_residuals, ax_residual = plt.subplots(1, 1, figsize=(12, 6))
            
            for i in range(num_simulations):
                color = colors[i]
                
                # Noisy data
                noisy_sim = noisy_results[i]
                if 'V1' in noisy_sim.observed_states and 'V2' in noisy_sim.observed_states:
                    V1_noisy = np.array(noisy_sim.observed_states['V1'])
                    V2_noisy = np.array(noisy_sim.observed_states['V2'])
                    if len(V1_noisy) == len(V2_noisy) and len(V1_noisy) > 0:
                        residual_noisy = V1_noisy + V2_noisy - self.V_total
                        time_steps = np.arange(len(residual_noisy))
                        mean_residual = np.mean(residual_noisy)
                        std_residual = np.std(residual_noisy)
                        
                        # Plot mean +/- 1.8 standard deviations
                        # ax_residual.fill_between(time_steps, 
                        #                        mean_residual - 1.3 * std_residual,
                        #                        mean_residual + 1.3 * std_residual,
                        #                        color=color, alpha=0.1)
                        ax_residual.plot(residual_noisy, color=color, linestyle='-', alpha=0.9, label=f"Noisy Sim {i+1}" if i == 0 else None)

                # Noiseless data
                noiseless_sim = noiseless_results[i]
                if 'V1' in noiseless_sim.observed_states and 'V2' in noiseless_sim.observed_states:
                    V1_noiseless = np.array(noiseless_sim.observed_states['V1'])
                    V2_noiseless = np.array(noiseless_sim.observed_states['V2'])
                    if len(V1_noiseless) == len(V2_noiseless) and len(V1_noiseless) > 0:
                        residual_noiseless = V1_noiseless + V2_noiseless - self.V_total
                        ax_residual.plot(residual_noiseless, color=color, linestyle='--', alpha=0.7, label=f"Noiseless Sim {i+1}" if i == 0 else None)
            
            ax_residual.set_title(f'Tank System Constraint Residual ($V_1 + V_2 - V_{{total}}$), $V_{{total}}={self.V_total}$')
            ax_residual.set_xlabel('Time step')
            ax_residual.set_ylabel('Residual ($m^3$)')
            ax_residual.axhline(0, color='black', linewidth=0.8, linestyle=':')
            ax_residual.legend(fontsize='small', loc='best')
            fig_residuals.tight_layout()
            plt.show(block=False)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        return 2 * (action - self.action_space['low']) / (self.action_space['high'] - self.action_space['low']) - 1

    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Convert normalized action back to original range."""
        return (action_norm + 1) * (self.action_space['high'] - self.action_space['low']) / 2 + self.action_space['low']

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation (states + setpoint) to [-1, 1] range."""
        return 2 * (obs - self.observation_space['low']) / (self.observation_space['high'] - self.observation_space['low']) - 1
    
    def _unnormalize_observation(self, obs_norm: np.ndarray) -> np.ndarray:
        """Convert normalized observation (states + setpoint) back to original range."""
        return (obs_norm + 1) * (self.observation_space['high'] - self.observation_space['low']) / 2 + self.observation_space['low']

    # No _unnormalize_disturbance needed as there are no disturbances

    def _format_results(
        self,
        observed_values_list: List[np.ndarray], # List of [V1, C1, V2, C2, V1_s] arrays
        disturbance_values_list: Dict[str, List[float]], # Empty for this model
        action_values_list: List[np.ndarray] # List of [F_in1, F_in2] arrays
    ) -> SimulationResult:
        
        # Transpose lists of arrays to dict of lists
        obs_states_dict = defaultdict(list)
        if observed_values_list:
            num_obs_vars = observed_values_list[0].shape[0]
            obs_keys = ['V1', 'C1', 'V2', 'C2', 'V1_s'] # Matches observation_space order
            for step_obs_array in observed_values_list:
                for i, key in enumerate(obs_keys):
                    if i < num_obs_vars: # safety check
                         obs_states_dict[key].append(step_obs_array[i])
        
        action_states_dict = defaultdict(list)
        if action_values_list:
            num_act_vars = action_values_list[0].shape[0]
            act_keys = ['F_in1', 'F_in2'] # Matches action_space order
            for step_act_array in action_values_list:
                for i, key in enumerate(act_keys):
                    if i < num_act_vars: # safety check
                        action_states_dict[key].append(step_act_array[i])
        
        # disturbance_states_dict will be empty from disturbance_values_list
        return SimulationResult(
            observed_states=dict(obs_states_dict), 
            disturbances=disturbance_values_list, # Already a dict of lists
            actions=dict(action_states_dict)
        )

# --- Main script execution ---
if __name__ == "__main__":
    np.random.seed(42)

    # Configuration
    config = SimulationConfig(
        n_simulations=1,  # Number of separate simulation runs
        T=100,            # Number of time steps per simulation
        tsim=100,        # Total simulation time (e.g., minutes) for T steps
        noise_percentage={ # Per-state noise if desired, or a single float
            'V1': 0.01, 'C1': 0.02, 'V2': 0.01, 'C2': 0.02 
            # pcgym expects single float or dict matching model states
            # This noise is applied by pcgym to its internal 'x' states.
        },
        V_total=100.0
    )

    simulator = TankSystemSimulator(config)
    
    # Run simulations
    noisy_sim_results, noiseless_sim_results = simulator.run_multiple_simulations()

    # Plot results (optional)
    simulator.plot_results(noisy_sim_results, noiseless_sim_results)
    simulator.plot_constraint_residuals(noisy_sim_results, noiseless_sim_results) # New call

    plt.show() # Ensure plots are displayed

    # Convert data to features and targets
    converter = TankSystemConverter()
    # Pass list of SimulationResult objects
    features, targets = converter.convert(noisy_sim_results) 
    noiseless_features, noiseless_targets = converter.convert(noiseless_sim_results)

    # Save to CSV
    features_df = pd.DataFrame(features, columns=['V1', 'C1', 'V2', 'C2', 'V1_s', 'F_in1', 'F_in2'])
    targets_df = pd.DataFrame(targets, columns=['V1', 'C1', 'V2', 'C2'])
    # For noiseless, perhaps save targets or full state representation
    noiseless_df = pd.DataFrame(noiseless_features, columns=['V1', 'C1', 'V2', 'C2', 'V1_s', 'F_in1', 'F_in2'])


    # features_df.to_csv('tank_system_features.csv', index=False)
    # targets_df.to_csv('tank_system_targets.csv', index=False)
    # noiseless_df.to_csv('tank_system_noiseless_features.csv', index=False)
    
    print("Tank system simulation data saved to CSV files.")