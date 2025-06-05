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
# from pcgym import make_env
# Fallback for local testing if pcgym is not fully set up or for a mock environment
try:
    from pcgym import make_env
    # from pcgym import BaseModel # If BaseModel is needed for type hinting
except ImportError:
    print("Warning: pcgym module not found. Using a mock 'make_env' for testing.")
    @dataclass
    class MockEnv:
        observation_space_low: np.ndarray
        observation_space_high: np.ndarray
        action_space_low: np.ndarray
        action_space_high: np.ndarray
        N: int
        x0: np.ndarray
        noise: bool
        noise_percentage: Union[float, Dict[str, float]]
        # model_states: List[str] # Not strictly needed by this simple mock
        # model_sp: List[str]     # Not strictly needed by this simple mock

        def __post_init__(self):
            self.current_step = 0
            self.internal_state = np.copy(self.x0) # x0 is [Th, Tc, Tw]

        def reset(self):
            self.current_step = 0
            self.internal_state = np.copy(self.x0)
            obs = np.copy(self.internal_state) # No SP in obs for this setup
            obs_norm = 2 * (obs - self.observation_space_low) / (self.observation_space_high - self.observation_space_low) - 1
            return obs_norm, {}

        def step(self, action_norm):
            action = (action_norm + 1) * (self.action_space_high - self.action_space_low) / 2 + self.action_space_low
            dt = 0.1 # mock time step 
            
            # internal_state is [Th, Tc, Tw], action is [Th_in, Tc_in]
            if len(self.internal_state) == 3 and len(action) == 2:
                 Th, Tc, Tw = self.internal_state
                 Th_in, Tc_in = action
                 
                 # Simplified mock dynamics (not the real model)
                 # These are placeholders and do not represent the heat exchanger physics
                 dTh_dt_mock = (Th_in - Th) * 0.1 - (Th - Tw) * 0.05 
                 dTc_dt_mock = (Tc_in - Tc) * 0.1 + (Tw - Tc) * 0.04
                 dTw_dt_mock = (Th - Tw) * 0.05 - (Tw - Tc) * 0.04 - (Tw - 298) * 0.001
                 
                 self.internal_state[0] += dTh_dt_mock * dt
                 self.internal_state[1] += dTc_dt_mock * dt
                 self.internal_state[2] += dTw_dt_mock * dt

                 # Basic clipping to keep within some bounds for mock
                 self.internal_state[0] = np.clip(self.internal_state[0], 280, 400) # Th
                 self.internal_state[1] = np.clip(self.internal_state[1], 280, 400) # Tc
                 self.internal_state[2] = np.clip(self.internal_state[2], 280, 400) # Tw
            
            current_state_to_observe = self.internal_state
            if self.noise:
                # Apply noise based on percentage of range or value
                if isinstance(self.noise_percentage, float):
                    noise_mag = self.noise_percentage * (self.observation_space_high - self.observation_space_low)
                else: # dict
                    noise_mag = np.array([self.noise_percentage.get(k, 0.01) * (self.observation_space_high[i] - self.observation_space_low[i]) 
                                          for i, k in enumerate(['Th', 'Tc', 'Tw'])]) # Assuming these keys match obs space order
                
                noise_val_state = np.random.normal(0, noise_mag / 3) # Approx std dev if mag is peak
                current_state_to_observe = self.internal_state + noise_val_state
            
            obs = np.copy(current_state_to_observe)
            obs_norm = 2 * (obs - self.observation_space_low) / (self.observation_space_high - self.observation_space_low) - 1
            
            self.current_step += 1
            done = self.current_step >= self.N
            return obs_norm, 0.0, done, False, {}

    def make_env(params: Dict):
        env = MockEnv(
            observation_space_low=params['o_space']['low'],
            observation_space_high=params['o_space']['high'],
            action_space_low=params['a_space']['low'],
            action_space_high=params['a_space']['high'],
            N=params['N'],
            x0=params['x0'],
            noise=params['noise'],
            noise_percentage=params['noise_percentage'],
        )
        return env

# --- Generic Simulation Helper Classes ---
@dataclass
class SimulationResult:
    observed_states: Dict[str, List[float]]
    disturbances: Dict[str, List[float]] # Kept for structure, will be empty
    actions: Dict[str, List[float]]

    def __iter__(self) -> Iterator[Dict[str, List[float]]]:
        return iter((self.observed_states, self.disturbances, self.actions))

@dataclass
class SimulationConfig:
    n_simulations: int
    T: int  
    tsim: float 
    noise_percentage: Union[float, Dict[str, float]] = 0.01
    # Heat Exchanger specific params (defaults from model, can be added here if needed for config)

class SimulationConverter(ABC):
    @abstractmethod
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        pass

# --- Heat Exchanger Specific Classes ---
class HeatExchangerConverter(SimulationConverter):
    def convert(self, data: List[SimulationResult]) -> Tuple[np.ndarray, np.ndarray]:
        obs_states_list = [res.observed_states for res in data]
        actions_list = [res.actions for res in data]
        
        combined_obs_states = defaultdict(list)
        for obs_sim in obs_states_list:
            for key, values in obs_sim.items():
                combined_obs_states[key].extend(values)
        
        combined_actions = defaultdict(list)
        for act_sim in actions_list:
            for key, values in act_sim.items():
                combined_actions[key].extend(values)

        # Physical states: Th, Tc, Tw
        # Actions (Inputs to model): Th_in, Tc_in
        feature_keys_obs = ['Th', 'Tc', 'Tw']
        feature_keys_act = ['Th_in', 'Tc_in']
        target_keys = ['Th', 'Tc', 'Tw'] # Predicting next state of physical variables

        if not combined_obs_states or not combined_actions:
             print("Error: No data to convert for heat exchanger.")
             return np.array([]), np.array([])
        
        try:
            num_total_timesteps = len(combined_obs_states[feature_keys_obs[0]])
            for key in feature_keys_obs:
                if len(combined_obs_states.get(key, [])) != num_total_timesteps:
                    raise ValueError(f"Mismatch in lengths for obs key {key} or key missing.")
            for key in feature_keys_act:
                if len(combined_actions.get(key, [])) != num_total_timesteps:
                     raise ValueError(f"Mismatch in lengths for act key {key} or key missing.")
        except (KeyError, IndexError) as e: # Catch if a primary key is missing
            print(f"Error: Missing key {e} or data for it in combined data during conversion.")
            return np.array([]), np.array([])
        except ValueError as e:
            print(f"Error: {e}")
            return np.array([]), np.array([])

        features_list = []
        for key in feature_keys_obs:
            features_list.append(np.array(combined_obs_states[key]).reshape(num_total_timesteps, 1))
        for key in feature_keys_act:
            features_list.append(np.array(combined_actions[key]).reshape(num_total_timesteps, 1))
        
        features = np.hstack(features_list)
        
        targets_list = []
        for key in target_keys:
            targets_list.append(np.array(combined_obs_states[key]).reshape(num_total_timesteps, 1))
        targets = np.hstack(targets_list)
        
        return features, targets

class HeatExchangerSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model_name = 'constrained_heat_exchanger' 

        # Inputs to model: Th_in, Tc_in
        self.action_space = {
            'low': np.array([323.15, 283.15]),    # Th_in (50C), Tc_in (10C) in K
            'high': np.array([373.15, 313.15])  # Th_in (100C), Tc_in (40C) in K
        }
        
        # Physical states: Th, Tc, Tw (all in K)
        # pcgym observation will be these states (normalized) if SP={}
        self.observation_space_keys = ['Th', 'Tc', 'Tw']
        self.observation_space = {
            'low': np.array([293.15, 288.15, 290.15]), # Th (20C), Tc (15C), Tw (17C)
            'high': np.array([383.15, 353.15, 373.15]) # Th (110C), Tc (80C), Tw (100C)
        }
        
        # Store model parameters for constraint residual calculation
        # Defaults from the provided model definition
        self.mh = 2.0     
        self.mc = 1.5     
        self.cp_h = 4.18  
        self.cp_c = 4.18  
        self.UA_amb = 5.0 / 1000 # Convert kW/K to kJ/(sK) if cp is in kJ - cp is kJ/kgK, UA_X is kW/K
                                 # Q terms in ODEs: UA*(T-T') -> kW. Denominators mh*cp_h -> kg/s * kJ/kgK = kW/K. This is inconsistent.
                                 # Let's assume all UAs are in kW/K as stated, and cp in kJ/kgK.
                                 # ODEs: dTh_dt = ( mh*cp_h*(Th_in-Th) [kW] - UA_h*(Th-Tw) [kW] ) / M_h_cp_h [kJ/K]
                                 # The model divides by (mh*cp_h) which is a rate [kW/K], not thermal mass [kJ/K].
                                 # This implies an effective time constant of 1s for the fluid.
                                 # For the residual, using parameters as given:
                                 # mh*cp_h and mc*cp_c are in kW/K (power per temp diff)
                                 # UA_amb is in kW/K
        self.T_amb = 298.0  
        # Note: The problem description of UA_h, UA_c, UA_amb as kW/K, and cp as kJ/kgK,
        # and the ODE structure: mh*cp_h*(Th_in - Th) - Q_h_to_w.
        # Here mh*cp_h has units kW/K. Th_in-Th is K. So mh*cp_h*(Th_in-Th) is kW. This is fine.
        # Q_h_to_w = UA_h * (Th-Tw) is kW. This is fine.
        # The denominator of dTh_dt is mh*cp_h [kW/K]. dTh_dt needs to be K/s.
        # So (kW) / (kW/K) = K. This is not K/s.
        # The model's ODEs as written imply the denominator (mh*cp_h for fluid, mw_cp for wall)
        # must actually be thermal capacitances (kJ/K), not capacitance rates.
        # E.g. M_h*cp_h, not mh*cp_h in denominator.
        # However, I will proceed with the model as literally given and assume the parameters
        # in the denominators are effective thermal masses with units kJ/K.
        # The problem states `mw_cp` as "Wall thermal mass x cp [kJ/K]", which is correct.
        # For the constraint calculation, `mh*cp_h` refers to `flow_rate * specific_heat`.

    # No generate_setpoints method as SP={} in env_params

    def generate_lhs_actions(self) -> np.ndarray: # Th_in, Tc_in
        sampler = qmc.LatinHypercube(d=len(self.action_space['low'])) 
        max_action_changes = self.config.T // 4 + 1
        samples = sampler.random(n=max_action_changes)
        scaled_samples = self.action_space['low'] + samples * (self.action_space['high'] - self.action_space['low'])
        
        actions_Th_in, actions_Tc_in = [], []
        num_changes = np.random.randint(0, max_action_changes)
        change_points = np.sort(np.random.choice(range(1, self.config.T), num_changes, replace=False))
        
        sample_idx = 0
        current_Th_in = scaled_samples[sample_idx, 0]
        current_Tc_in = scaled_samples[sample_idx, 1]
        
        for t_step in range(self.config.T):
            if len(change_points) > 0 and t_step == change_points[0]:
                sample_idx = (sample_idx + 1) % max_action_changes
                current_Th_in = scaled_samples[sample_idx, 0]
                current_Tc_in = scaled_samples[sample_idx, 1]
                change_points = change_points[1:]
            actions_Th_in.append(current_Th_in)
            actions_Tc_in.append(current_Tc_in)
            
        return np.array([actions_Th_in, actions_Tc_in]).T 

    def generate_x0(self) -> np.ndarray: # Th, Tc, Tw
        # Initial conditions ensuring some temperature gradient, e.g. Tc_init < Tw_init < Th_init
        # And within defined observation space bounds
        low = self.observation_space['low']
        high = self.observation_space['high']

        Th_init = np.random.uniform( (low[0]+high[0])/2 , high[0] ) 
        Tc_init = np.random.uniform( low[1], (low[1]+high[1])/2 )
        Tw_init = np.random.uniform( Tc_init, Th_init ) # Tw between Tc and Th

        # Clip to ensure they are strictly within overall bounds
        Th_init = np.clip(Th_init, low[0], high[0])
        Tc_init = np.clip(Tc_init, low[1], high[1])
        Tw_init = np.clip(Tw_init, low[2], high[2])
        # Further ensure Tw is between Tc and Th after clipping
        Tw_init = np.clip(Tw_init, Tc_init, Th_init) if Tc_init < Th_init else np.random.uniform(low[2], high[2])


        return np.array([Th_init, Tc_init, Tw_init])

    def simulate(self) -> Tuple[SimulationResult, SimulationResult]:
        unnorm_action_sequence = self.generate_lhs_actions() 
        action_sequence_normalized = self._normalize_action(unnorm_action_sequence) 
        x0_physical = self.generate_x0() 
        
        env_params = {
            'N': self.config.T,         
            'tsim': self.config.tsim,   
            'SP': {}, # No setpoints appended to observation by pcgym
            'o_space': self.observation_space, 
            'a_space': self.action_space,     
            'x0': x0_physical, # Initial physical states [Th, Tc, Tw]
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

        # obs_norm from pcgym will be [norm_Th, norm_Tc, norm_Tw]
        obs_norm, _ = env.reset() 
        noiseless_obs_norm, _ = noiseless_env.reset()
        
        for step_num in range(self.config.T):
            action_normalized_current_step = action_sequence_normalized[step_num, :]
            
            obs_norm, _, done, _, info = env.step(action_normalized_current_step)
            noiseless_obs_norm, _, _, _, _ = noiseless_env.step(action_normalized_current_step)
            
            obs_unnorm = self._unnormalize_observation(obs_norm) # [Th, Tc, Tw]
            noiseless_obs_unnorm = self._unnormalize_observation(noiseless_obs_norm)
            action_unnorm = unnorm_action_sequence[step_num, :] # [Th_in, Tc_in]
            
            obs_list_noisy.append(obs_unnorm)
            actions_list_noisy.append(action_unnorm)
            obs_list_noiseless.append(noiseless_obs_unnorm)
            actions_list_noiseless.append(action_unnorm) 

            if done:
                break
        
        empty_disturbances_values = {} # No disturbances in this model
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
        if not noisy_results: return
        num_sims = len(noisy_results)
        colors = plt.cm.viridis(np.linspace(0, 1, num_sims))

        # States: Th, Tc, Tw
        state_keys_to_plot = self.observation_space_keys # ['Th', 'Tc', 'Tw']
        state_titles = ['Hot Fluid Temp $T_h$ (K)', 'Cold Fluid Temp $T_c$ (K)', 'Wall Temp $T_w$ (K)']
        
        fig_states, axs_states = plt.subplots(len(state_keys_to_plot), 1, figsize=(10, 4 * len(state_keys_to_plot)), squeeze=False)

        for i in range(num_sims):
            color, n_sim, nl_sim = colors[i], noisy_results[i], noiseless_results[i]
            lbl_sfx = f"Sim {i+1}"
            for j, key in enumerate(state_keys_to_plot):
                if key in n_sim.observed_states and n_sim.observed_states[key] and \
                   key in nl_sim.observed_states and nl_sim.observed_states[key]:
                    axs_states[j, 0].plot(n_sim.observed_states[key], c=color, alpha=0.8, label=f"Noisy {lbl_sfx}" if i==0 else None)
                    axs_states[j, 0].plot(nl_sim.observed_states[key], c=color, ls='--', alpha=0.8, label=f"Noiseless {lbl_sfx}" if i==0 else None)
        
        for j, title in enumerate(state_titles):
            axs_states[j, 0].set_title(title)
            axs_states[j, 0].set_xlabel("Time step")
            axs_states[j, 0].set_ylabel("Temperature (K)")
            if num_sims > 0 : axs_states[j,0].legend(fontsize='small', loc='best')
        
        fig_states.tight_layout()
        plt.show(block=False)

        # Actions: Th_in, Tc_in
        action_keys = ['Th_in', 'Tc_in']
        action_titles = ['Inlet Hot Fluid Temp $T_{h,in}$ (K)', 'Inlet Cold Fluid Temp $T_{c,in}$ (K)']
        fig_actions, axs_actions = plt.subplots(len(action_keys), 1, figsize=(10, 4 * len(action_keys)), squeeze=False)
        
        for i in range(num_sims):
            color, n_sim = colors[i], noisy_results[i]
            lbl_sfx = f"Sim {i+1}"
            for j, key in enumerate(action_keys):
                if key in n_sim.actions and n_sim.actions[key]:
                    axs_actions[j, 0].plot(n_sim.actions[key], c=color, alpha=0.8, label=f"{lbl_sfx}" if i==0 else None)

        for j, title in enumerate(action_titles):
            axs_actions[j, 0].set_title(title)
            axs_actions[j, 0].set_xlabel("Time step")
            axs_actions[j, 0].set_ylabel("Temperature (K)")
            if num_sims > 0: axs_actions[j,0].legend(fontsize='small', loc='best')
        
        fig_actions.tight_layout()
        plt.show(block=False)

    def plot_constraint_residuals(self, noisy_results: List[SimulationResult], noiseless_results: List[SimulationResult]):
        """Plots the residual of the energy balance constraint."""
        if not noisy_results: return
        num_sims = len(noisy_results)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_sims))
        fig_residuals, ax_residual = plt.subplots(1, 1, figsize=(12, 6))
        
        for i in range(num_sims):
            color = colors[i]
            # Noisy data
            noisy_sim = noisy_results[i]
            if all(k in noisy_sim.observed_states for k in ['Th', 'Tc', 'Tw']) and \
               all(k in noisy_sim.actions for k in ['Th_in', 'Tc_in']):
                
                Th_arr_n = np.array(noisy_sim.observed_states['Th'])
                Tc_arr_n = np.array(noisy_sim.observed_states['Tc'])
                Tw_arr_n = np.array(noisy_sim.observed_states['Tw'])
                Th_in_arr_n = np.array(noisy_sim.actions['Th_in'])
                Tc_in_arr_n = np.array(noisy_sim.actions['Tc_in'])

                if not all(len(arr) == len(Th_arr_n) and len(arr)>0 for arr in [Tc_arr_n, Tw_arr_n, Th_in_arr_n, Tc_in_arr_n]):
                    print(f"Skipping noisy residual plot for sim {i+1} due to data length mismatch.")
                else:
                    LHS_n = self.mh * self.cp_h * (Th_in_arr_n - Th_arr_n)
                    RHS_n = self.mc * self.cp_c * (Tc_arr_n - Tc_in_arr_n) + self.UA_amb * (Tw_arr_n - self.T_amb)
                    residual_n = LHS_n - RHS_n
                    ax_residual.plot(residual_n, color=color, linestyle='-', alpha=0.7, label=f"Noisy Sim {i+1}" if i == 0 else None)

            # Noiseless data
            noiseless_sim = noiseless_results[i]
            if all(k in noiseless_sim.observed_states for k in ['Th', 'Tc', 'Tw']) and \
               all(k in noiseless_sim.actions for k in ['Th_in', 'Tc_in']):

                Th_arr_nl = np.array(noiseless_sim.observed_states['Th'])
                Tc_arr_nl = np.array(noiseless_sim.observed_states['Tc'])
                Tw_arr_nl = np.array(noiseless_sim.observed_states['Tw'])
                Th_in_arr_nl = np.array(noiseless_sim.actions['Th_in'])
                Tc_in_arr_nl = np.array(noiseless_sim.actions['Tc_in'])
                
                if not all(len(arr) == len(Th_arr_nl) and len(arr)>0 for arr in [Tc_arr_nl, Tw_arr_nl, Th_in_arr_nl, Tc_in_arr_nl]):
                     print(f"Skipping noiseless residual plot for sim {i+1} due to data length mismatch.")
                else:
                    LHS_nl = self.mh * self.cp_h * (Th_in_arr_nl - Th_arr_nl)
                    RHS_nl = self.mc * self.cp_c * (Tc_arr_nl - Tc_in_arr_nl) + self.UA_amb * (Tw_arr_nl - self.T_amb)
                    residual_nl = LHS_nl - RHS_nl
                    ax_residual.plot(residual_nl, color=color, linestyle='--', alpha=0.7, label=f"Noiseless Sim {i+1}" if i == 0 else None)
        
        ax_residual.set_title('Heat Exchanger Energy Balance Residual')
        ax_residual.set_xlabel('Time step')
        ax_residual.set_ylabel('Residual (kW)') # Assuming units are consistent as kW
        ax_residual.axhline(0, color='black', linewidth=0.8, linestyle=':')
        if num_sims > 0: ax_residual.legend(fontsize='small', loc='best')
        fig_residuals.tight_layout()
        plt.show(block=False)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        return 2 * (action - self.action_space['low']) / (self.action_space['high'] - self.action_space['low']) - 1

    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return (action_norm + 1) * (self.action_space['high'] - self.action_space['low']) / 2 + self.action_space['low']

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        return 2 * (obs - self.observation_space['low']) / (self.observation_space['high'] - self.observation_space['low']) - 1
    
    def _unnormalize_observation(self, obs_norm: np.ndarray) -> np.ndarray:
        # obs_norm is [norm_Th, norm_Tc, norm_Tw]
        # self.observation_space['low'] and ['high'] must match this order and content
        return (obs_norm + 1) * (self.observation_space['high'] - self.observation_space['low']) / 2 + self.observation_space['low']

    def _format_results(
        self,
        observed_values_list: List[np.ndarray], # List of [Th, Tc, Tw] arrays for each step
        disturbance_values_list: Dict[str, List[float]], # Empty
        action_values_list: List[np.ndarray] # List of [Th_in, Tc_in] arrays for each step
    ) -> SimulationResult:
        obs_states_dict = defaultdict(list)
        if observed_values_list:
            # self.observation_space_keys = ['Th', 'Tc', 'Tw']
            for step_obs_array in observed_values_list: 
                for i, key in enumerate(self.observation_space_keys):
                    if i < len(step_obs_array): 
                         obs_states_dict[key].append(step_obs_array[i])
        
        action_states_dict = defaultdict(list)
        if action_values_list:
            act_keys = ['Th_in', 'Tc_in'] 
            for step_act_array in action_values_list: 
                for i, key in enumerate(act_keys):
                    if i < len(step_act_array): 
                        action_states_dict[key].append(step_act_array[i])
        
        return SimulationResult(dict(obs_states_dict), disturbance_values_list, dict(action_states_dict))

# --- Main script execution ---
if __name__ == "__main__":
    np.random.seed(777)

    config = SimulationConfig(
        n_simulations=2,  
        T=250,            
        tsim=50.0,        # Simulation time (e.g., seconds)
        noise_percentage={ # Per-state noise for Th, Tc, Tw
            'Th': 0.005, 'Tc': 0.005, 'Tw': 0.002 
            # This would be % of range if pcgym uses it like that, or absolute if value based
        } 
        # Or a single float e.g., noise_percentage=0.01
    )

    simulator = HeatExchangerSimulator(config)
    noisy_sim_results, noiseless_sim_results = simulator.run_multiple_simulations()

    simulator.plot_results(noisy_sim_results, noiseless_sim_results)
    simulator.plot_constraint_residuals(noisy_sim_results, noiseless_sim_results) 
    plt.show() 

    converter = HeatExchangerConverter()
    features, targets = converter.convert(noisy_sim_results) 
    noiseless_features, _ = converter.convert(noiseless_sim_results) # Targets usually from noisy

    # if features.size > 0 and targets.size > 0:
    #     feature_column_names = ['Th', 'Tc', 'Tw', 'Th_in', 'Tc_in']
    #     target_column_names = ['Th', 'Tc', 'Tw']
    #     features_df = pd.DataFrame(features, columns=feature_column_names)
    #     targets_df = pd.DataFrame(targets, columns=target_column_names)
        
    #     features_df.to_csv('heat_exchanger_features.csv', index=False)
    #     targets_df.to_csv('heat_exchanger_targets.csv', index=False)
    #     print("Heat exchanger feature and target data saved.")

    # if noiseless_features.size > 0:
    #     noiseless_df = pd.DataFrame(noiseless_features, columns=feature_column_names)
    #     noiseless_df.to_csv('heat_exchanger_noiseless_features.csv', index=False)
    #     print("Heat exchanger noiseless feature data saved.")
    
    print("Heat exchanger simulation finished.")