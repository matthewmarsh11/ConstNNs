import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Dummy data
X = torch.tensor([[1.0]], dtype=torch.float32)
y_true = torch.tensor([[2.0]], dtype=torch.float32)

# Define simple NN
class SimpleNet(nn.Module):
    def __init__(self, w1_init, w2_init):
        super().__init__()
        self.fc1_weight = nn.Parameter(torch.tensor([[w1_init]], dtype=torch.float32))
        self.fc1_bias = nn.Parameter(torch.tensor([w2_init], dtype=torch.float32))
        self.out_mu = nn.Linear(1, 1)
        self.out_var = nn.Linear(1, 1)

    def forward(self, x):
        hidden = F.relu(x @ self.fc1_weight + self.fc1_bias)
        mu = self.out_mu(hidden)
        var = F.softplus(self.out_var(hidden)) + 1e-6
        return mu, var

# Loss functions
def mse_loss_fn(mu, y):
    return F.mse_loss(mu, y)

def gnll_loss_fn(mu, var, y):
    var_clamped = torch.clamp(var, min=1e-6) # Clamp variance for stability
    return 0.5 * torch.log(2 * torch.pi * var_clamped) + (y - mu)**2 / (2 * var_clamped)

# Weight sweep
w1_vals = np.linspace(-4, 4, 50)
w2_vals = np.linspace(-4, 4, 50)
W1_grid, W2_grid = np.meshgrid(w1_vals, w2_vals)

# Loss containers
MSE_vals = np.zeros_like(W1_grid)
GNLL_vals = np.zeros_like(W1_grid)

# Evaluate losses across weight grid
for i in range(W1_grid.shape[0]):
    for j in range(W1_grid.shape[1]):
        net = SimpleNet(W1_grid[i, j], W2_grid[i, j])
        with torch.no_grad():
            mu, var = net(X)
            MSE_vals[i, j] = mse_loss_fn(mu, y_true).item()
            GNLL_vals[i, j] = gnll_loss_fn(mu, var, y_true).item()

# Simulate optimization trajectories
def simulate_gradient_descent(loss_surface, w1_grid_vals, w2_grid_vals, start_point_coords, lr=0.1, steps=50):
    trajectory = np.zeros((steps, 2))
    trajectory[0] = start_point_coords
    dw1 = w1_grid_vals[1] - w1_grid_vals[0]
    dw2 = w2_grid_vals[1] - w2_grid_vals[0]
    for i in range(1, steps):
        current_w1, current_w2 = trajectory[i-1]
        idx_w1 = np.clip(np.searchsorted(w1_grid_vals, current_w1) -1, 1, len(w1_grid_vals)-2)
        idx_w2 = np.clip(np.searchsorted(w2_grid_vals, current_w2) -1, 1, len(w2_grid_vals)-2)
        grad_w1 = (loss_surface[idx_w2, idx_w1+1] - loss_surface[idx_w2, idx_w1-1]) / (2 * dw1)
        grad_w2 = (loss_surface[idx_w2+1, idx_w1] - loss_surface[idx_w2-1, idx_w1]) / (2 * dw2)
        trajectory[i, 0] = current_w1 - lr * grad_w1
        trajectory[i, 1] = current_w2 - lr * grad_w2
        trajectory[i, 0] = np.clip(trajectory[i, 0], w1_grid_vals.min(), w1_grid_vals.max())
        trajectory[i, 1] = np.clip(trajectory[i, 1], w2_grid_vals.min(), w2_grid_vals.max())
        if i > 5 and np.linalg.norm(trajectory[i] - trajectory[i-5]) < 0.01:
            trajectory[i:] = trajectory[i]
            return trajectory, True, i
    return trajectory, False, steps -1

start_points = [
    [-3, -3], [3, 3], [-3, 3], [3, -3],
    [0, 0], [1, -2], [-2, 1], [-1,-3.5], [3.5,1]
]

plt.rcParams.update({'font.size': 10})

# --- Figure 1: Weight Space Loss Landscapes with Gradient Directions ---
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(15, 6.5)) # Adjusted size slightly
fig1.suptitle("Loss Landscapes & Gradient Fields in Weight Space\n(W1 controls fc1_weight, W2 controls fc1_bias)", fontsize=16)

# Calculate gradients for weight space
spacing_w1 = w1_vals[1] - w1_vals[0]
spacing_w2 = w2_vals[1] - w2_vals[0]
grad_W2_mse, grad_W1_mse = np.gradient(MSE_vals, spacing_w2, spacing_w1) # Note: order of output from np.gradient matches axis order
grad_W2_gnll, grad_W1_gnll = np.gradient(GNLL_vals, spacing_w2, spacing_w1)

# Quiver skip factor
skip_factor = 1 # Show ~10 arrows per dimension for a 50x50 grid
skip = (slice(None, None, skip_factor), slice(None, None, skip_factor))

# MSE contour plot (Weight Space) with gradients
mse_levels_ws = np.linspace(np.percentile(MSE_vals,1), np.percentile(MSE_vals,99), 20)
contour1a = ax1a.contourf(W1_grid, W2_grid, np.clip(MSE_vals, mse_levels_ws.min(), mse_levels_ws.max()), levels=mse_levels_ws, cmap='viridis', alpha=0.9, extend='both')
ax1a.contour(W1_grid, W2_grid, MSE_vals, levels=mse_levels_ws, colors='k', linewidths=0.5, alpha=0.3)
cb1a = fig1.colorbar(contour1a, ax=ax1a, label="MSE Loss", shrink=0.9)
ax1a.quiver(W1_grid[skip], W2_grid[skip], -grad_W1_mse[skip], -grad_W2_mse[skip],
            color='white', alpha=0.7, scale=None, scale_units='xy', angles='xy', width=0.003, headwidth=4)
ax1a.set_title("MSE Loss Landscape & Gradients")
ax1a.set_xlabel("w1_init (fc1_weight)")
ax1a.set_ylabel("w2_init (fc1_bias)")

# GNLL contour plot (Weight Space) with gradients
gnll_clip_min_ws = np.percentile(GNLL_vals[np.isfinite(GNLL_vals)],1)
gnll_clip_max_ws = np.percentile(GNLL_vals[np.isfinite(GNLL_vals)],95)
gnll_levels_ws = np.linspace(gnll_clip_min_ws, gnll_clip_max_ws, 20)
contour1b = ax1b.contourf(W1_grid, W2_grid, np.clip(GNLL_vals, gnll_levels_ws.min(), gnll_levels_ws.max()), levels=gnll_levels_ws, cmap='inferno', alpha=0.9, extend='both')
ax1b.contour(W1_grid, W2_grid, GNLL_vals, levels=gnll_levels_ws, colors='k', linewidths=0.5, alpha=0.3)
cb1b = fig1.colorbar(contour1b, ax=ax1b, label="GNLL Loss", shrink=0.9)
ax1b.quiver(W1_grid[skip], W2_grid[skip], -grad_W1_gnll[skip], -grad_W2_gnll[skip],
            color='white', alpha=0.7, scale=None, scale_units='xy', angles='xy', width=0.003, headwidth=4) # Adjust scale as needed, e.g. scale=50
ax1b.set_title("GNLL Loss Landscape & Gradients")
ax1b.set_xlabel("w1_init (fc1_weight)")
ax1b.set_ylabel("w2_init (fc1_bias)")
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()


# --- Figure 2: Output Space Loss Landscapes with Gradient Directions ---
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle("Loss Landscapes & Gradient Fields in Output Space", fontsize=16)

# MSE in Output Space (Loss vs. Predicted Mean μ)
mu_range_output_mse = np.linspace(y_true.item() - 3, y_true.item() + 3, 200)
mse_output_landscape = np.array([(mu_val - y_true.item())**2 for mu_val in mu_range_output_mse])
ax2a.plot(mu_range_output_mse, mse_output_landscape, 'royalblue', linewidth=2.5, alpha=0.7)
ax2a.fill_between(mu_range_output_mse, mse_output_landscape, alpha=0.2, color='royalblue')
ax2a.set_title("MSE Loss vs. Predicted Mean (μ)")
ax2a.set_xlabel("Predicted Mean (μ)")
ax2a.set_ylabel("MSE Loss")
ax2a.axvline(x=y_true.item(), color='red', linestyle='--', linewidth=2, label=f'True Mean (μ={y_true.item():.1f})')
ax2a.grid(True, alpha=0.4)
min_mse_loss = 0
ax2a.plot(y_true.item(), min_mse_loss, 'ro', markersize=8, label='Global Minimum')
# For 1D MSE plot, gradient arrows are less conventional than for 2D.
# Adding a few indicative arrows manually if desired:
skip_1d_mse = len(mu_range_output_mse) // 10
for idx in range(skip_1d_mse // 2, len(mu_range_output_mse), skip_1d_mse):
    mu_val = mu_range_output_mse[idx]
    loss_val = mse_output_landscape[idx]
    grad_mu_mse = 2 * (mu_val - y_true.item())
    # Arrow points in negative gradient direction
    ax2a.arrow(mu_val, loss_val, -grad_mu_mse * 0.05, 0, # Scale arrow length, dx, dy=0
               head_width=0.5, head_length=0.1, fc='gray', ec='gray', alpha=0.8)
ax2a.legend(fontsize=8)

# GNLL in Output Space (Loss vs. μ and σ²)
mu_range_gnll_output = np.linspace(y_true.item() - 3, y_true.item() + 4, 50)
var_range_gnll_output = np.linspace(0.05, 5, 50)
MU_output, VAR_output = np.meshgrid(mu_range_gnll_output, var_range_gnll_output)
GNLL_output_space_vals = np.zeros_like(MU_output)
# Calculate analytical gradients for GNLL output space
grad_mu_gnll_output = np.zeros_like(MU_output)
grad_var_gnll_output = np.zeros_like(VAR_output)

for i in range(MU_output.shape[0]):
    for j in range(MU_output.shape[1]):
        mu_ij, var_ij = MU_output[i,j], VAR_output[i,j]
        if var_ij <= 1e-6: # Avoid division by zero / log(0)
            GNLL_output_space_vals[i,j] = np.inf
            grad_mu_gnll_output[i,j] = 0
            grad_var_gnll_output[i,j] = 0
        else:
            GNLL_output_space_vals[i,j] = 0.5 * np.log(2 * np.pi * var_ij) + (y_true.item() - mu_ij)**2 / (2 * var_ij)
            grad_mu_gnll_output[i,j] = (mu_ij - y_true.item()) / var_ij
            grad_var_gnll_output[i,j] = 0.5 / var_ij - (y_true.item() - mu_ij)**2 / (2 * var_ij**2)


min_loss_gnll_out = np.nanmin(GNLL_output_space_vals[np.isfinite(GNLL_output_space_vals)])
max_loss_gnll_out = np.nanpercentile(GNLL_output_space_vals[np.isfinite(GNLL_output_space_vals)], 95)
dynamic_gnll_levels_out = np.linspace(min_loss_gnll_out, max_loss_gnll_out, 20)

contour2b = ax2b.contourf(MU_output, VAR_output, np.clip(GNLL_output_space_vals, min_loss_gnll_out, max_loss_gnll_out),
                          levels=dynamic_gnll_levels_out, cmap='inferno', extend='both')
ax2b.contour(MU_output, VAR_output, GNLL_output_space_vals, levels=dynamic_gnll_levels_out, colors='k', linewidths=0.5, alpha=0.3)
cb2b = fig2.colorbar(contour2b, ax=ax2b, label="GNLL Loss", shrink=0.9)
ax2b.quiver(MU_output[skip], VAR_output[skip], -grad_mu_gnll_output[skip], -grad_var_gnll_output[skip],
            color='white', alpha=0.7, scale=None, scale_units='xy', angles='xy', width=0.003, headwidth=4) # Adjust scale, e.g. scale=10
ax2b.set_title("GNLL Loss Landscape & Gradients")
ax2b.set_xlabel("Predicted Mean (μ)")
ax2b.set_ylabel("Predicted Variance (σ²)")
ax2b.axvline(x=y_true.item(), color='lime', linestyle='--', linewidth=1.5, label=f'True Mean (μ={y_true.item():.1f})')
ax2b.plot(y_true.item(), np.min(var_range_gnll_output), 'go', markersize=8, alpha=0.7, label='Approx. Target (μ=true, σ²→0)')
ax2b.legend(fontsize=8)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()


# --- Figure 3: Weight Space Optimization Trajectories --- (NO CHANGES)
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
fig3.suptitle("Optimization Trajectories in Weight Space", fontsize=16)
ax3a.set_title("MSE Trajectories")
ax3a.set_xlabel("w1_init (fc1_weight)")
ax3a.set_ylabel("w2_init (fc1_bias)")
ax3a.set_xlim(w1_vals.min(), w1_vals.max())
ax3a.set_ylim(w2_vals.min(), w2_vals.max())
ax3a.grid(True, linestyle=':', alpha=0.7)
ax3b.set_title("GNLL Trajectories")
ax3b.set_xlabel("w1_init (fc1_weight)")
ax3b.grid(True, linestyle=':', alpha=0.7)
for i, sp_coords in enumerate(start_points):
    color = plt.cm.tab10(i % 10)
    traj_mse_w, _, conv_step_mse_w = simulate_gradient_descent(MSE_vals, w1_vals, w2_vals, sp_coords, lr=0.2, steps=100)
    ax3a.plot(traj_mse_w[:conv_step_mse_w+1, 0], traj_mse_w[:conv_step_mse_w+1, 1], '-', color=color, linewidth=1.5)
    ax3a.plot(sp_coords[0], sp_coords[1], 'o', color=color, markersize=6, markeredgecolor='k')
    ax3a.plot(traj_mse_w[conv_step_mse_w, 0], traj_mse_w[conv_step_mse_w, 1], 'x', color=color, markersize=7, markeredgewidth=1.5)
    traj_gnll_w, _, conv_step_gnll_w = simulate_gradient_descent(GNLL_vals, w1_vals, w2_vals, sp_coords, lr=0.05, steps=150)
    ax3b.plot(traj_gnll_w[:conv_step_gnll_w+1, 0], traj_gnll_w[:conv_step_gnll_w+1, 1], '-', color=color, linewidth=1.5)
    ax3b.plot(sp_coords[0], sp_coords[1], 'o', color=color, markersize=6, markeredgecolor='k')
    ax3b.plot(traj_gnll_w[conv_step_gnll_w, 0], traj_gnll_w[conv_step_gnll_w, 1], 'x', color=color, markersize=7, markeredgewidth=1.5)
legend_elements_traj = [
    plt.Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=6, markeredgecolor='k', label='Start'),
    plt.Line2D([0], [0], marker='x', color='gray', linestyle='', markersize=7, markeredgewidth=1.5, label='End'),
    plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5, label='Path')
]
fig3.legend(handles=legend_elements_traj, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()

# --- Figure 4: Output Space Optimization Trajectories --- (NO CHANGES)
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 7))
fig4.suptitle("Optimization Trajectories in Output Space", fontsize=16)
ax4a.set_title("MSE Trajectories (μ vs. Loss)")
ax4a.set_xlabel("Predicted Mean (μ)")
ax4a.set_ylabel("MSE Loss")
ax4a.grid(True, linestyle=':', alpha=0.7)
ax4a.set_xlim(mu_range_output_mse.min(), mu_range_output_mse.max())
ax4a.set_ylim(min_mse_loss -1, np.max(mse_output_landscape) * 1.1 if np.max(mse_output_landscape) > 0 else 10) # ensure positive ylim
ax4b.set_title("GNLL Trajectories (μ vs. σ²)")
ax4b.set_xlabel("Predicted Mean (μ)")
ax4b.set_ylabel("Predicted Variance (σ²)")
ax4b.grid(True, linestyle=':', alpha=0.7)
ax4b.set_xlim(mu_range_gnll_output.min(), mu_range_gnll_output.max())
ax4b.set_ylim(var_range_gnll_output.min(), var_range_gnll_output.max())
for i, sp_coords in enumerate(start_points):
    color = plt.cm.tab10(i % 10)
    traj_mse_w, _, conv_step_mse_w = simulate_gradient_descent(MSE_vals, w1_vals, w2_vals, sp_coords, lr=0.2, steps=100)
    output_mu_traj_mse, output_loss_traj_mse = [], []
    for k_step in range(conv_step_mse_w + 1):
        w1_k, w2_k = traj_mse_w[k_step, 0], traj_mse_w[k_step, 1]
        net_k = SimpleNet(w1_k, w2_k)
        with torch.no_grad(): mu_k, _ = net_k(X); loss_k = mse_loss_fn(mu_k, y_true).item()
        output_mu_traj_mse.append(mu_k.item()); output_loss_traj_mse.append(loss_k)
    ax4a.plot(output_mu_traj_mse, output_loss_traj_mse, '-', color=color, linewidth=1.5)
    ax4a.plot(output_mu_traj_mse[0], output_loss_traj_mse[0], 'o', color=color, markersize=6, markeredgecolor='k')
    ax4a.plot(output_mu_traj_mse[-1], output_loss_traj_mse[-1], 'x', color=color, markersize=7, markeredgewidth=1.5)
    traj_gnll_w, _, conv_step_gnll_w = simulate_gradient_descent(GNLL_vals, w1_vals, w2_vals, sp_coords, lr=0.05, steps=150)
    output_mu_traj_gnll, output_var_traj_gnll = [], []
    for k_step in range(conv_step_gnll_w + 1):
        w1_k, w2_k = traj_gnll_w[k_step, 0], traj_gnll_w[k_step, 1]
        net_k = SimpleNet(w1_k, w2_k)
        with torch.no_grad(): mu_k, var_k = net_k(X)
        output_mu_traj_gnll.append(mu_k.item()); output_var_traj_gnll.append(var_k.item())
    ax4b.plot(output_mu_traj_gnll, output_var_traj_gnll, '-', color=color, linewidth=1.5)
    ax4b.plot(output_mu_traj_gnll[0], output_var_traj_gnll[0], 'o', color=color, markersize=6, markeredgecolor='k')
    ax4b.plot(output_mu_traj_gnll[-1], output_var_traj_gnll[-1], 'x', color=color, markersize=7, markeredgewidth=1.5)
fig4.legend(handles=legend_elements_traj, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()