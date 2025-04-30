import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, multivariate_normal

# Parameters
epsilon = np.array([1.5, 3.0])  # Anisotropic tolerances
alpha = 0.05
k = 2
chi2_crit = chi2.ppf(1 - alpha, k)

# Scaling matrix Σ_ε
Sigma_epsilon = np.diag(epsilon**2 / chi2_crit)

# Original covariance Σ_Z (e.g., from prior)
Sigma_Z_prior = np.array([[2.0, 1.2], [1.2, 1.5]])

# Adjusted covariance Σ_Z (to satisfy Σ_Z ⪯ Σ_ε)
eigvals_prior, eigvecs_prior = np.linalg.eigh(Sigma_Z_prior)
eigvals_adj = np.minimum(eigvals_prior, np.diag(Sigma_epsilon))
Sigma_Z_adj = eigvecs_prior @ np.diag(eigvals_adj) @ eigvecs_prior.T

# Plot covariance ellipses
theta = np.linspace(0, 2*np.pi, 100)
ellipse_prior = eigvecs_prior @ np.diag(np.sqrt(eigvals_prior)) @ np.array([np.cos(theta), np.sin(theta)])
ellipse_adj = eigvecs_prior @ np.diag(np.sqrt(eigvals_adj)) @ np.array([np.cos(theta), np.sin(theta)])
ellipse_epsilon = np.sqrt(Sigma_epsilon) @ np.array([np.cos(theta), np.sin(theta)])

plt.figure(figsize=(10, 6))
plt.plot(ellipse_prior[0], ellipse_prior[1], 'b--', label='Original Σ_Z')
plt.plot(ellipse_adj[0], ellipse_adj[1], 'r-', label='Adjusted Σ_Z')
plt.plot(ellipse_epsilon[0], ellipse_epsilon[1], 'g--', label='Constraint Σ_ε')
plt.xlabel('Z₁')
plt.ylabel('Z₂')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()