import numpy as np
import matplotlib.pyplot as plt

# True target
y_true = 2.0

# Grid setup
mu_vals = np.linspace(0, 4, 600)
var_vals = np.linspace(0.01, 4, 600)
MU, VAR = np.meshgrid(mu_vals, var_vals)

# Compute NLL
NLL = 0.5 * np.log(2 * np.pi * VAR) + ((y_true - MU)**2) / (2 * VAR)

# Contour levels
min_loss = np.min(NLL)
max_loss = np.percentile(NLL, 97)
levels = np.linspace(min_loss, max_loss, 400)


# Plot
fig, ax = plt.subplots(figsize=(12, 8))
contour = ax.contourf(MU, VAR, NLL, levels=levels, cmap='inferno', extend='both')

# Colorbar
cbar = plt.colorbar(contour, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label('NLL Loss', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Mark important points
ax.plot(y_true, 0.5, 'g*', markersize=18, label='Good Prediction')
ax.plot(1, 1.8, 'o', color='cyan', markersize=18, label='Local Optima Trap')

# Gradient field
skip = (slice(None, None, 40), slice(None, None, 40))
grad_mu = (MU - y_true) / VAR
grad_var = -0.5 / VAR + ((y_true - MU)**2) / (2 * VAR**2)
ax.quiver(MU[skip], VAR[skip], -grad_mu[skip], -grad_var[skip],
          color='white', alpha=0.6, scale=10, width=0.002, label='−∇NLL')


# Labels and legend
ax.set_xlabel('Predicted Mean (μ)', fontsize=13)
ax.set_ylabel('Predicted Variance (σ²)', fontsize=13)
ax.set_title('Gaussian NLL Loss Surface with Optimisation Paths', fontsize=15)
ax.legend(fontsize=10)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.show()