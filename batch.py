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
        """Makes SimulationResult iterable, yielding (observed_states, disturbances, actions)"""
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

class BatchConverter(SimulationConverter):
    def convert(self, data: List[Tuple[Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the process simulation data into features and targets

        Args:
            data (List[Tuple[Dict, Dict]]): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        obs_states = [obs for obs, _ in data]
        actions = [act for _, act in data]

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
    
class BatchSimulator():
    def __init__(
        self,
        config
    ):
        """
        Initialize the Batch simulator.
        
        Args:
            T (int): Number of time steps
            tsim (int): Simulation time period
            noise_percentage (float): Noise level for simulation
        """
        self.config = config
        
        # Define spaces
        self.action_space = {
            'low': np.array([250]),
            'high': np.array([450])
        }
        
        self.observation_space = {
            'low': np.array([0, 0, 0, 200, 100]),
            'high': np.array([1, 1, 1, 350, 150])
        }
        
        self.uncertainty_percentages = { 
            'rho': 0.01, 'Cp': 0.01, 
            'EAoverR_AB': 0.01, 'k0_A': 0.01, 
            'UA': 0.01, 'mdelH_AB': 0.01,
            'mdelH_BC': 0.01,
            'EAoverR_BC': 0.01, 'k0_B': 0.01,
        }
        
        self.uncertainty_space = {
            'low': np.array([0]*9),
            'high': np.array([1]*9)
        }

    def generate_setpoints(self) -> Dict[str, List[float]]:
        """Generate random setpoints for the simulation."""
        num_changes = np.random.randint(0, self.config.T // 4)
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
        setpoints = []
        current_setpoint = np.random.rand()
        
        for t in range(self.config.T):
            if len(change_points) > 0 and t == change_points[0]:
                current_setpoint = np.random.rand()
                change_points = change_points[1:]
            setpoints.append(current_setpoint)
        
        return {'Ca': setpoints}


    def generate_lhs_actions(self) -> List[float]:
        """Generate action sequence using Latin Hypercube Sampling with step changes."""
        # Create LHS sampler for actions
        sampler = qmc.LatinHypercube(d=1)
        
        # Generate base action values using LHS
        max_actions = self.config.T // 15  # Maximum number of possible changes
        samples = sampler.random(n=max_actions)
        
        # Scale samples to action space
        action_range = self.action_space['high'] - self.action_space['low']
        action_samples = self.action_space['low'] + samples * action_range
        
        # Generate change points for actions
        num_changes = np.random.randint(0, max_actions)
        change_points = np.sort(np.random.choice(range(1, self.config.tsim), num_changes, replace=False))
        
        # Initialize action sequence
        actions = []
        action_idx = 0
        current_action = action_samples[action_idx] + (np.random.rand() - 0.5) * 0.1  # Small variation
        
        # Generate action sequence with step changes
        for t in range(self.config.T):
            if len(change_points) > 0 and t == change_points[0]:
                action_idx = (action_idx + 1) % len(action_samples)
                current_action = action_samples[action_idx] + (np.random.rand() - 0.5) * 0.1
                change_points = change_points[1:]
            actions.append(current_action)
        
        return np.array(actions)
        
        # return self._normalize_action(np.array(actions))
    
    def generate_x0(self) -> np.ndarray:
        """Generate initial state for the simulation as a random deviation from the midpoint of each observation bound."""
        midpoints = (self.observation_space['high'] + self.observation_space['low']) / 2
        deviations = (np.random.rand(*self.observation_space['low'].shape) - 0.5) * (self.observation_space['high'] - self.observation_space['low']) * 0.1
        return midpoints + deviations
    
    def simulate(self) -> SimulationResult:
        """Run a single simulation with LHS-generated step changes for actions and disturbances."""
        setpoints = self.generate_setpoints()
        # Generate complete action sequence
        unnorm_action_sequence = self.generate_lhs_actions()
        action_sequence = self._normalize_action(unnorm_action_sequence)
        x0 = self.generate_x0()
        
        env_params = {
            'N': self.config.T,
            'tsim': self.config.tsim,
            'SP': setpoints,
            'o_space': self.observation_space,
            'a_space': self.action_space,
            'x0': x0,
            'model': 'batch',
            'noise': True,
            'noise_percentage': self.config.noise_percentage,
            'normalise_o': True,
            'normalise_a': True,
            'custom_model': None,
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
            
            
            # Unnormalize values
            obs_unnorm = self._unnormalize_observation(obs)
            actions_unnorm = self._unnormalize_action(action)
            
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs)
            
            
            # Store results
            observed_states.append(obs_unnorm)
            noiseless_observed_states.append(noiseless_obs_unnorm)
            actions.append(actions_unnorm)
            

            # Increment step counter
            step += 1
    
        
        return self._format_results(observed_states,actions), self._format_results(noiseless_observed_states, actions)

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
        - States (Ca, Cb, Cc, T, V, Ca_s) with both noisy and noiseless data
        - Actions (Tc, Fin) from noisy data only
        - Disturbances (Ti, Caf) from noisy data only
        
        Args:
            noisy_results (List[SimulationResult]): List of noisy simulation results
            noiseless_results (List[SimulationResult]): List of noiseless simulation results
        """
        # Create three separate figures for states, actions, and disturbances
        fig_states, axs_states = plt.subplots(3, 2, figsize=(15, 12))
        fig_act, axs_act = plt.subplots(2, 1, figsize=(10, 8))

        # Get a color for each simulation pair
        colors = plt.cm.tab10(np.linspace(0, 1, len(noisy_results)))
        
        # Plot states (with noisy/noiseless comparison)
        for i, ((noisy, noiseless), color) in enumerate(zip(zip(noisy_results, noiseless_results), colors)):
            for j, state in enumerate(['Ca', 'Cb', 'Cc', 'T', 'Ca_s']):
                row, col = divmod(j, 2)
                axs_states[row, col].plot(noisy.observed_states[state], 
                                          label=f'Simulation {i+1}', 
                                          color=color,
                                          alpha=0.7)
                axs_states[row, col].plot(noiseless.observed_states[state], 
                                          label=f'Noiseless {i+1}', 
                                          color=color,
                                          linestyle='--', 
                                          alpha=0.7)
        
            # Plot actions (noisy only)
            for j, action in enumerate(['Tc']):
                axs_act[j].plot(noisy.actions[action], 
                            label=f'Simulation {i+1} - {action}', 
                            color=color,
                            alpha=0.7,
                            linestyle='-')
            
        
        # Set titles and labels for states
        state_titles = ['Concentration of A (Ca)', 'Concentration of B (Cb)', 'Concentration of C (Cc)', 
                        'Temperature (T)', 'Setpoint of Ca (Ca_s)']
        for j, title in enumerate(state_titles):
            row, col = divmod(j, 2)
            axs_states[row, col].set_title(title)
            axs_states[row, col].set_xlabel('Time')
            axs_states[row, col].set_ylabel(title.split('(')[1][:-1])
        
        # Set titles and labels for actions
        action_titles = ['Cooling Jacket Temperature (Tc)']
        for j, title in enumerate(action_titles):
            axs_act[j].set_title(title)
            axs_act[j].set_xlabel('Time')
            axs_act[j].set_ylabel(title.split('(')[1][:-1])

        
        # Add legends to all plots
        for ax in axs_states.flat:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        for ax in axs_act:
            ax.legend(loc='upper right', fontsize='small', ncol=2)
        
        # Adjust layouts
        for fig in [fig_states, fig_act]:
            fig.tight_layout()
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


    def _format_results(
        self,
        observed_states: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> SimulationResult:
        """Format the simulation results into a structured container."""
        obs_states = {
            'Ca': [state[0] for state in observed_states],
            'Cb': [state[1] for state in observed_states],
            'Cc': [state[2] for state in observed_states],
            'T': [state[3] for state in observed_states],
            'Ca_s': [state[4] for state in observed_states],
        }

        
        action_states = {
            'Tc': [state[0] for state in actions],
        }
        
        return SimulationResult(obs_states, action_states)

config = SimulationConfig(n_simulations=10, T=300, tsim=300, noise_percentage={'Ca': 0.02, 'Cb': 0.02, 'Cc': 0.02, 'T': 0.005})
simulator = BatchSimulator(config)

# # Run multiple simulations

simulation_results, noiseless_results = simulator.run_multiple_simulations()

simulator.plot_results(simulation_results, noiseless_results)

converter = BatchConverter()
features, targets = converter.convert(simulation_results)
noiseless_results, _ = converter.convert(noiseless_results)

# # Save the features and targets to CSV files
# features_df = pd.DataFrame(features)
# targets_df = pd.DataFrame(targets)
# noiseless_results_df = pd.DataFrame(noiseless_results)

# features_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_batch_features.csv', index=False)
# targets_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_batch_targets.csv', index=False)
# noiseless_results_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/datasets/small_batch_noiseless_results.csv', index=False)