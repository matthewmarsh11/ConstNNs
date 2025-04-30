import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def plot_gaussian_adjustment():
    # Parameters
    np.random.seed(42)
    mu_p = np.array([0, 0])        # Mean of P(y)
    Sigma_p = np.array([[2, 1],    # Covariance of P(y)
                       [1, 2]])
    B = np.array([[1, 0],          # Constraint matrix
                  [0, 1]])
    epsilon = 3.0                  # Perturbation tolerance
    alpha = 0.05                   # Violation probability
    k = B.shape[0]                 # Rank of B

    # Chi-squared critical value
    chi2_crit = chi2.ppf(1 - alpha, df=k)
    
    # Adjusted covariance calculation (closed-form solution)
    C_p = B @ Sigma_p @ B.T
    theta = (1/(epsilon**2/chi2_crit) - 1/np.max(np.linalg.eigvals(C_p))) / np.linalg.norm(B)**2
    Sigma_q_inv = np.linalg.inv(Sigma_p) + theta * B.T @ B
    Sigma_q = np.linalg.inv(Sigma_q_inv)
    mu_q = mu_p - Sigma_p @ B.T @ np.linalg.inv(B @ Sigma_p @ B.T) @ (B @ mu_p)

    # Generate points for plotting
    theta_grid = np.linspace(0, 2*np.pi, 100)
    
    # Plot in y-space
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Original and adjusted distributions in y-space
    plt.subplot(121)
    for Sigma, color, label in [(Sigma_p, 'blue', 'P(y)'), 
                               (Sigma_q, 'red', 'Q(y)')]:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        axes = eigvecs @ np.diag(np.sqrt(eigvals * chi2.ppf(0.95, 2)))
        x = axes[0,0] * np.cos(theta_grid) + axes[0,1] * np.sin(theta_grid)
        y = axes[1,0] * np.cos(theta_grid) + axes[1,1] * np.sin(theta_grid)
        plt.plot(x + mu_p[0], y + mu_p[1], color=color, label=label)
    
    plt.scatter(*mu_p, c='blue', marker='x', label='Mean of P(y)')
    plt.scatter(*mu_q, c='red', marker='x', label='Mean of Q(y)')
    plt.title("Original (P) vs Adjusted (Q) in y-space")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    # Plot 2: Transformed z-space with constraint
    plt.subplot(122)
    for Sigma, color, label in [(C_p, 'blue', 'P(z)'), 
                               (B @ Sigma_q @ B.T, 'red', 'Q(z)')]:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        axes = eigvecs @ np.diag(np.sqrt(eigvals * chi2.ppf(0.95, 2)))
        x = axes[0,0] * np.cos(theta_grid) + axes[0,1] * np.sin(theta_grid)
        y = axes[1,0] * np.cos(theta_grid) + axes[1,1] * np.sin(theta_grid)
        plt.plot(x, y, color=color, label=label)
    
    # Plot epsilon-ball constraint
    plt.gca().add_patch(plt.Circle((0, 0), epsilon, fill=False, 
                                 linestyle='--', color='gray', 
                                 label=f'ε-ball (ε={epsilon})'))
    plt.title("Transformed z-space with Constraint")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

plot_gaussian_adjustment()