import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Iterator
from dataclasses import dataclass
from tqdm import tqdm
from scipy.stats import qmc
import pandas as pd
from collections import defaultdict
from abc import abstractmethod
from pcgym import make_env

np.random.seed(42)

@dataclass
class SimulationResult:
    """Container for simulation results."""
    observed_states: Dict[str, List[float]]
    actions: Dict[str, List[float]]
    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        """Makes SimulationResult iterable, yielding (observed_states, actions)"""
        return iter((self.observed_states, self.actions))

@dataclass
class SimulationConfig:
    """Configuration for simulation data collection"""
    n_simulations: int
    T: int
    tsim: int
    noise_percentage: Union[float, Dict] = 0.01

class SimulationConverter():
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data) -> Tuple[np.array, np.array]:
        """Convert output of simulation and return features and targets"""
        pass

class BiofilmConverter(SimulationConverter):
    def convert(self, data: List[Tuple[Dict, Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the process simulation data into features and targets

        Args:
            data (List[Tuple[Dict, Dict, Dict]]): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        obs_states = [obs for obs, _ in data]
        actions = [act for _, act in data]
        # The setpoint means nothing in this model as there is no defined 
        # relationship between the setpoint and the process

        for obs in obs_states:
            if 'Ca_s' in obs:
                del obs['Ca_s']
                
        # Combine obs_states into a single dictionary
        combined_obs_states = defaultdict(list)
        for obs in obs_states:
            for key, value in obs.items():
                combined_obs_states[key].append(value)
        
        # Convert lists to numpy arrays
        for key in combined_obs_states:
            combined_obs_states[key] = np.array(combined_obs_states[key])
        
            
        combined_actions = defaultdict(list)
        for act in actions:
            for key, value in act.items():
                combined_actions[key].append(value)
        
        for key in combined_actions:
            combined_actions[key] = np.array(combined_actions[key])    
        
        combined_features = {**combined_obs_states, **combined_actions}
        targets = {k: combined_obs_states[k] for k in ['Ca', 'Cb', 'Cc', 'T']}
        aggregated_data = defaultdict(list)
        aggregated_targets = defaultdict(list)
         
        for d, value in combined_features.items():
            aggregated_data[d].append(value)
            aggregated_data[d] = aggregated_data[d][0]
        
        aggregated_data['simulation_no'] = np.array([[i + 1] * len(aggregated_data['Ca'][0]) for i in range(len(aggregated_data['Ca']))])
        
        for d, value in targets.items():
            aggregated_targets[d].append(value)
            aggregated_targets[d] = aggregated_targets[d][0]
        
        aggregated_data = dict(aggregated_data)
        
        features = np.array(list(aggregated_data.values()))
        features = features.transpose(1, 2, 0)
        features = features.reshape(features.shape[0] * features.shape[1], -1)

        targets = np.array(list(aggregated_targets.values()))
        targets = targets.transpose(1, 2, 0)
        targets = targets.reshape(targets.shape[0] * targets.shape[1], -1)
        
        return features, targets
    
class BiofilmSimulator():
    def __init__(
        self,
        config
    ):
        """
        Initialize the CSTR simulator.
        
        Args:
            T (int): Number of time steps
            tsim (int): Simulation time period
            noise_percentage (float): Noise level for simulation
        """
        self.config = config
        
        # Define spaces
        self.action_space = {
            'low': np.array([0, 0, 0.5, 0.5, 0.05]),
            'high': np.array([10, 30, 1, 1, 1])
        }
        
        self.observation_space = {
            'low': np.array([0] * 16),
            'high': np.array([1] * 16)
        }
                
        self.uncertainty_percentages = { 
            'V': 0.01, 'Va': 0.01, 'Kla': 0.01, 'm': 0.01,
            'eq_exponent': 0.01, 'O_air': 0.01, 'vm_1': 0.01,
            'vm_2': 0.01, 'K1': 0.01, 'K2': 0.01, 'KO_1': 0.01,
            'KO_2': 0.01
        
        }
        
        self.uncertainty_space = {
            'low': np.array([0]*9),
            'high': np.array([1]*9)
        }


    def generate_lhs_actions(self) -> np.ndarray:
        """Generate full 5D action sequence using Latin Hypercube Sampling with step changes for each dimension."""
        action_dim = len(self.action_space['low'])
        max_changes = self.config.T // 4  # max number of changes

        # Latin Hypercube samples for each action dimension
        sampler = qmc.LatinHypercube(d=action_dim)
        samples = sampler.random(n=max_changes)

        # Scale samples to action space
        action_range = self.action_space['high'] - self.action_space['low']
        action_samples = self.action_space['low'] + samples * action_range

        # Generate independent change points for each action dimension
        change_points = {
            i: np.sort(np.random.choice(range(1, self.config.T), np.random.randint(1, max_changes), replace=False))
            for i in range(action_dim)
        }

        actions = np.zeros((self.config.T, action_dim))
        current_actions = action_samples[0]

        # Initialize tracking indices
        action_indices = {i: 0 for i in range(action_dim)}

        for t in range(self.config.T):
            for i in range(action_dim):
                if len(change_points[i]) > 0 and t == change_points[i][0]:
                    action_indices[i] = (action_indices[i] + 1) % max_changes
                    current_actions[i] = action_samples[action_indices[i], i] + (np.random.rand() - 0.5) * 0.1
                    change_points[i] = change_points[i][1:]
            actions[t] = current_actions

        return actions
        
        # return self._normalize_action(np.array(actions))
    
    def generate_x0(self) -> np.ndarray:
        """Generate initial state for the simulation as a random deviation from the midpoint of each observation bound."""
        midpoints = (self.observation_space['high'] + self.observation_space['low']) / 2
        deviations = (np.random.rand(*self.observation_space['low'].shape) - 0.5) * (self.observation_space['high'] - self.observation_space['low']) * 0.1
        return midpoints + deviations
    
    def simulate(self) -> SimulationResult:
        """Run a single simulation with LHS-generated step changes for actions and disturbances."""
        # Generate complete action sequence
        unnorm_action_sequence = self.generate_lhs_actions()
        action_sequence = unnorm_action_sequence.copy()
        # action_sequence = self._normalize_action(unnorm_action_sequence)
        x0 = self.generate_x0()
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0,
            'model': 'biofilm_reactor',
            'reward_states': np.array(['O_A']),
            'maximise_reward': True,
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': False,
            'normalise_a': False,
            'custom_model': None,
            # 'integration_method': 'jax',
            # 'uncertainty_percentages': self.uncertainty_percentages,
            # 'uncertainty_bounds': self.uncertainty_space,
            # 'distribution': 'normal',
        }
        
        # Create environments
        env = make_env(env_params)
        noiseless_env_params = env_params.copy()
        noiseless_env_params['noise'] = False
        noiseless_env_params['noise_percentage'] = 0
        noiseless_env = make_env(noiseless_env_params)
        
        # Initialize simulation variables
        observed_states = []
        actions = []
        noiseless_observed_states = []
        noiseless_disturbance_values = []
        
        obs, _ = env.reset()
        noiseless_obs, _ = noiseless_env.reset()
        done = False
        step = 0
        
        # Simulation loop
        while not done:
            # Get current action from sequence
            action = action_sequence[step]
            
            obs, _, done, _, info = env.step(action)
            noiseless_obs, _, _, _, _ = noiseless_env.step(action)
            
            # Split and process observations
            # obs = obs[:5]
            
            # noiseless_uncertain_params = noiseless_obs[8:]
            # noiseless_obs = noiseless_obs[:
            
            # Unnormalize values
            # obs_unnorm = self._unnormalize_observation(obs)
            obs_unnorm = obs
            # disturbance_unnorm = self._unnormalize_disturbance(disturbance, disturbance_space)
            # actions_unnorm = self._unnormalize_action(action)
            actions_unnorm = action
            # noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs)
            noiseless_obs_unnorm = noiseless_obs
            # noiseless_disturbance_unnorm = self._unnormalize_disturbance(noiseless_disturbance, disturbance_space)
            
            # Store results
            observed_states.append(obs_unnorm)
    
            noiseless_observed_states.append(noiseless_obs_unnorm)
            actions.append(actions_unnorm)
            

            # Increment step counter
            step += 1
    
        
        return self._format_results(observed_states, actions), self._format_results(noiseless_observed_states, actions)

    def run_multiple_simulations(self) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        """
        Run multiple simulations and return both noisy and noiseless results.
        
        Returns:
            Tuple[List[SimulationResult], List[SimulationResult]]: Lists of (noisy_results, noiseless_results)
        """
        noisy_results = []
        noiseless_results = []
        
        for _ in tqdm(range(self.config.n_simulations), desc="Running simulations"):
            noisy_sim, noiseless_sim = self.simulate()
            noisy_results.append(noisy_sim)
            noiseless_results.append(noiseless_sim)
        return noisy_results, noiseless_results
    

    def plot_results(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]) -> None:
        """
        Plot all simulation results:
        - States (S1_1, S2_1, S3_1, O_1, S1_2, S2_2, S3_2, O_2, S1_3, S2_3, S3_3, O_3, S1_A, S2_A, S3_A, O_A) with both noisy and noiseless data
        - Actions (F, Fr, S1_F, S2_F, S3_F) from noisy data only

        
        Args:
            noisy_results (List[SimulationResult]): List of noisy simulation results
            noiseless_results (List[SimulationResult]): List of noiseless simulation results
        """
        # Create separate figures for states and actions
        fig_states, axs_states = plt.subplots(4, 4, figsize=(15, 12))
        axs_states = axs_states.reshape(4, 4)  # Ensure proper shape for indexing
        fig_act, axs_act = plt.subplots(3, 2, figsize=(10, 15))
        axs_act = axs_act.flatten()  # Flatten to make indexing simpler
        
        # Get a color for each simulation pair
        colors = plt.cm.tab10(np.linspace(0, 1, len(noisy_results)))
        
        # Plot states (with noisy/noiseless comparison)
        for i, ((noisy, noiseless), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            for j, state in enumerate(['S1_1', 'S2_1', 'S3_1', 'O_1', 'S1_2', 'S2_2', 'S3_2', 'O_2', 'S1_3', 'S2_3', 'S3_3', 'O_3', 'S1_A', 'S2_A', 'S3_A', 'O_A']):
                row, col = divmod(j, 4)  # Calculate correct row and column
                axs_states[row, col].plot(noisy.observed_states[state], 
                                        label=f'Simulation {i+1}', 
                                        color=color,
                                        alpha=0.7)
                axs_states[row, col].plot(noiseless.observed_states[state], 
                                        label=f'Noiseless {i+1}', 
                                        color=color,
                                        linestyle='--', 
                                        alpha=0.7)
                axs_states[row, col].set_title(f'{state}')
                axs_states[row, col].grid(True, alpha=0.3)
        
            # Plot actions (noisy only)
            for j, action in enumerate(['F', 'Fr', 'S1_F', 'S2_F', 'S3_F']):
                if j < len(axs_act):  # Make sure we don't exceed the array bounds
                    axs_act[j].plot(noisy.actions[action], 
                            label=f'Simulation {i+1}', 
                            color=color,
                            alpha=0.7)
                    axs_act[j].set_title(f'{action}')
                    axs_act[j].grid(True, alpha=0.3)
        
        # Add legends to all plots
        for ax in axs_states.flat:
            ax.legend(loc='upper right', fontsize='small')
        for j in range(min(len(['F', 'Fr', 'S1_F', 'S2_F', 'S3_F']), len(axs_act))):
            axs_act[j].legend(loc='upper right', fontsize='small')
        
        # Adjust layouts
        fig_states.tight_layout()
        fig_act.tight_layout()
        
        plt.show()
    
    def _normalize_action(self, action: Union[float, np.ndarray]) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        return 2 * (action - self.action_space['low']) / (
            self.action_space['high'] - self.action_space['low']
        ) - 1

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [-1, 1] range."""
        return 2 * (obs - self.observation_space['low']) / (
            self.observation_space['high'] - self.observation_space['low']
        ) - 1
    
    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action back to original range."""
        return (action + 1) * (
            self.action_space['high'] - self.action_space['low']
        ) / 2 + self.action_space['low']

    def _unnormalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert normalized observation back to original range."""
        return (obs + 1) * (
            self.observation_space['high'] - self.observation_space['low']
        ) / 2 + self.observation_space['low']

    def _unnormalize_disturbance(self, disturbance: np.ndarray, space: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert normalized disturbance back to original range."""
        return (disturbance + 1) * (space['high'] - space['low']) / 2 + space['low']

    def _format_results(
        self,
        observed_states: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> SimulationResult:
        """Format the simulation results into a structured container."""
        obs_states = {
            'S1_1': [state[0] for state in observed_states],
            'S2_1': [state[1] for state in observed_states],
            'S3_1': [state[2] for state in observed_states],
            'O_1': [state[3] for state in observed_states],
            'S1_2': [state[4] for state in observed_states],
            'S2_2': [state[5] for state in observed_states],
            'S3_2': [state[6] for state in observed_states],
            'O_2': [state[7] for state in observed_states],
            'S1_3': [state[8] for state in observed_states],
            'S2_3': [state[9] for state in observed_states],
            'S3_3': [state[10] for state in observed_states],
            'O_3': [state[11] for state in observed_states],
            'S1_A': [state[12] for state in observed_states],
            'S2_A': [state[13] for state in observed_states],
            'S3_A': [state[14] for state in observed_states],
            'O_A': [state[15] for state in observed_states],          
        }
        
        
        action_states = {
            'F': [action[0] for action in actions],
            'Fr': [action[1] for action in actions],
            'S1_F': [action[2] for action in actions],
            'S2_F': [action[3] for action in actions],
            'S3_F': [action[4] for action in actions],
        }
        
        return SimulationResult(obs_states, action_states)

config = SimulationConfig(n_simulations=1, T=100, tsim=100, noise_percentage=0.01)
simulator = BiofilmSimulator(config)

# # # Run multiple simulations

simulation_results, noiseless_results = simulator.run_multiple_simulations()

simulator.plot_results(simulation_results, noiseless_results)

# converter = CSTRConverter()
# features, targets = converter.convert(simulation_results)
# noiseless_results, _ = converter.convert(noiseless_results)

# # Save the features and targets to CSV files
# features_df = pd.DataFrame(features)
# targets_df = pd.DataFrame(targets)
# noiseless_results_df = pd.DataFrame(noiseless_results)

# features_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_cstr_features.csv', index=False)
# targets_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_cstr_targets.csv', index=False)
# noiseless_results_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_cstr_noiseless_results.csv', index=False)