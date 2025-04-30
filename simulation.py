from cstr import *
from utils import *
import pandas as pd

# Simulate the CSTR 10 times, with 5000 timesteps over 1000 second period
CSTR_Config = SimulationConfig(n_simulations=1,
                                T = 101,
                                tsim = 500,
                                noise_percentage=0.01,
                            ) 

class SimulationConverter():
    """Converts the simulation data into features and targets to be used in the model"""
    @abstractmethod
    def convert(self, data) -> Tuple[np.array, np.array]:
        """Convert output of simulation and return features and targets"""
        pass

class CSTRConverter(SimulationConverter):
    def convert(self, data: List[Tuple[Dict, Dict, Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the process simulation data into features and targets

        Args:
            data (List[Tuple[Dict, Dict, Dict]]): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        obs_states = [obs for obs, _, _ in data]
        disturbances = [dist for _, dist, _ in data]
        actions = [act for _, _, act in data]
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
        
        combined_disturbances = defaultdict(list)
        for dist in disturbances:
            for key, value in dist.items():
                combined_disturbances[key].append(value)
        
        for key in combined_disturbances:
            combined_disturbances[key] = np.array(combined_disturbances[key])
            
        combined_actions = defaultdict(list)
        for act in actions:
            for key, value in act.items():
                combined_actions[key].append(value)
        
        for key in combined_actions:
            combined_actions[key] = np.array(combined_actions[key])    
        
        combined_features = {**combined_obs_states, **combined_disturbances, **combined_actions}
        targets = {k: combined_obs_states[k] for k in ['Ca', 'T']}
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

# simulator = CSTRSimulator(CSTR_Config)
# simulator = CSTRSimulator(CSTR_Config)

# simulation_results, noiseless_sim = simulator.run_multiple_simulations()

# # Plot the output of the Simulation
# # simulator.plot_results(simulation_results, noiseless_sim)
# # converter = BioprocessConverter()
# converter = CSTRConverter()
# features, targets = converter.convert(simulation_results)
# noiseless_results, _ = converter.convert(noiseless_sim)

# # Save the features and targets to CSV files
# features_df = pd.DataFrame(features)
# targets_df = pd.DataFrame(targets)
# noiseless_results_df = pd.DataFrame(noiseless_results)

# features_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/small_cstr_features.csv', index=False)
# targets_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/small_cstr_targets.csv', index=False)
# noiseless_results_df.to_csv('/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/ConstNNs/small_cstr_noiseless_results.csv', index=False)
# # Define a preliminary training configuration for the model
# # Data processing uses an initial lookback region of 5 timesteps to predict 1 in the future 
# # with an 80% train test split and a batch size of 4
# training_config = TrainingConfig(
#     batch_size = 4,
#     num_epochs = 50,
#     learning_rate = 0.001,
#     time_step = 10,
#     horizon = 5,
#     weight_decay = 0.01,
#     factor = 0.8,
#     patience = 10,
#     delta = 0.1,
#     train_test_split = 0.8,
#     device = 'cuda' if torch.cuda.is_available() else 'cpu',
# )

# data_processor = DataProcessor(training_config, features, targets)
# (train_loader, test_loader, val_loader, X_train, X_test, X_val, y_train, y_test, y_val, X, y) = data_processor.prepare_data()