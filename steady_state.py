import numpy as np

class steady_state_cstr():
    # Same parameters as your original model
    q: float = 100              
    V: float = 100              
    rho: float = 1000           
    C: float = 0.239            
    deltaHr1: float = -5e4       
    EA1_over_R: float = 8750     
    k01: float = 7.2e10         
    deltaHr2: float = -3e4       
    EA2_over_R: float = 9000     
    k02: float = 1.0e10         
    UA: float = 5e4             
    Ti: float = 350             
    Caf: float = 1              
    
    def __post_init__(self):
        self.states = ["Ca", "Cb", "Cc", "T"]
        self.inputs = ["Tc"]
        self.disturbances = ["Ti", "Caf"]

    def steady_state_equations(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Returns steady-state algebraic equations (should equal zero at steady-state)
        """
        ca, cb, cc, T = x[0], x[1], x[2], x[3]
        
        # Handle inputs
        if u.shape == (1,):
            Tc = u[0]
        else:
            Tc, self.Ti, self.Caf = u[0], u[1], u[2]
        
        # Calculate reaction rates
        r1 = self.k01 * np.exp(-self.EA1_over_R / T) * ca
        r2 = self.k02 * np.exp(-self.EA2_over_R / T) * cb
        
        # Steady-state material balances (set derivatives = 0)
        eq1 = (self.q/self.V)*(self.Caf - ca) - r1  # = 0
        eq2 = (self.q/self.V)*(0 - cb) + 2*r1 - r2   # = 0
        eq3 = (self.q/self.V)*(0 - cc) + r2          # = 0
        
        # Steady-state energy balance
        heat_gen = (-self.deltaHr1 * r1) + (-self.deltaHr2 * r2)
        eq4 = (self.q/self.V)*(self.Ti - T) + heat_gen/(self.rho * self.C) + (self.UA/(self.rho * self.C * self.V))*(Tc - T)  # = 0
        
        return np.array([eq1, eq2, eq3, eq4])

    def solve_steady_state(self, u: np.ndarray, initial_guess=None):
        """
        Solve for steady-state given inputs u
        """
        from scipy.optimize import fsolve
        
        if initial_guess is None:
            initial_guess = [0.1, 0.1, 0.1, 400]  # [Ca, Cb, Cc, T]
        
        # Solve the nonlinear system
        solution = fsolve(self.steady_state_equations, initial_guess, args=(u,))
        return solution

def generate_steady_state_data(model, n_samples=1000):
    """
    Generate steady-state data by varying inputs and solving for equilibrium
    """
    # Define input ranges
    Tc_range = np.linspace(300, 500, 50)      # Coolant temperature
    Ti_range = np.linspace(320, 380, 20)      # Inlet temperature  
    Caf_range = np.linspace(0.5, 2.0, 20)    # Feed concentration
    
    features = []
    targets = []
    
    for Tc in Tc_range:
        for Ti in Ti_range:
            for Caf in Caf_range:
                # Set model parameters
                model.Ti = Ti
                model.Caf = Caf
                
                # Solve steady-state
                u = np.array([Tc])
                try:
                    steady_state = model.solve_steady_state(u)
                    
                    # Store data
                    features.append([Ti, Caf, Tc])
                    targets.append(steady_state)  # [Ca, Cb, Cc, T]
                    
                except:
                    continue  # Skip if no solution found
    
    return np.array(features), np.array(targets)

def simulate_to_steady_state(dynamic_model, u, t_final=1000, rtol=1e-6):
    """
    Simulate dynamic model until steady-state is reached
    """
    from scipy.integrate import solve_ivp
    
    # Initial conditions
    x0 = [0.1, 0.1, 0.1, 350]  # [Ca, Cb, Cc, T]
    
    # Simulate with constant inputs
    def model_func(t, x):
        return dynamic_model(x, u)
    
    # Long simulation time
    t_span = [0, t_final]
    t_eval = np.linspace(0, t_final, 10000)
    
    sol = solve_ivp(model_func, t_span, x0, t_eval=t_eval, rtol=rtol)
    
    # Return final state (steady-state)
    return sol.y[:, -1]

# Create steady-state model
ss_model = steady_state_cstr()

# Generate data
features, targets = generate_steady_state_data(ss_model)

import matplotlib.pyplot as plt

# Create subplots for each feature
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Plot each feature
axes[0].hist(features[:, 0], bins=50)
axes[0].set_title('Inlet Temperature (Ti)')
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Frequency')

axes[1].hist(features[:, 1], bins=50)
axes[1].set_title('Feed Concentration (Caf)')
axes[1].set_xlabel('Concentration')
axes[1].set_ylabel('Frequency')

axes[2].hist(features[:, 2], bins=50)
axes[2].set_title('Coolant Temperature (Tc)')
axes[2].set_xlabel('Temperature (K)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()