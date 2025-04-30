import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# plt.rcParams.update({
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "font.family": "serif",  # Use a serif font
#     "font.serif": ["Computer Modern"],  # Default LaTeX font
#     "axes.labelsize": 14,  # Set font size for axes labels
#     "font.size": 12,  # Set global font size
#     "legend.fontsize": 12,  # Set legend font size
#     "xtick.labelsize": 12,  # Set x-axis tick size
#     "ytick.labelsize": 12,  # Set y-axis tick size
# })

# === Define distributions ===
# Original (from model)
mu_1 = 5
sigma = 0.5  # shared variance (no change after projection)

# Projected (mean projected into constraint zone, same sigma)
mu_2 = 3  # Assume projection moves the mean to constraint center

# Constraint center and tolerance (soft equality on mean)
c = 3          # target value for equality constraint on mean
epsilon = 0.5    # tolerance around c
delta = 0.05     # (unused here, but related to confidence)

# Range for plotting
x = np.linspace(1, 8, 1000)
pdf_1 = norm.pdf(x, mu_1, sigma)
pdf_2 = norm.pdf(x, mu_2, sigma)

# === Plot ===
plt.figure(figsize=(10, 5))

# Plot both distributions
plt.plot(x, pdf_1, label=f'Unconstrained Output: $P(y)$', linewidth=2)
plt.plot(x, pdf_2, label=f'Projected Output: $Q(y)$', linewidth=2, linestyle='--')

# Shade the constraint band for the **mean**
# plt.axvspan(c - epsilon, c + epsilon, color='gray', alpha=0.2, label='Constraint on Mean')

# Draw a vertical line at c (target mean)
plt.axvline(c, color='red', linestyle='--', linewidth=4, label='Constraint on Mean')
plt.text(c + 0.05, max(pdf_1.max(), pdf_2.max()) * 0.5, r'$Ax + B\mu_Q = b$', color = 'red', rotation=90, fontweight='bold', fontsize=14)

# Draw arrow from original mean to projected mean (same sigma)
peak1_x, peak1_y = mu_1, norm.pdf(mu_1, mu_1, sigma)
peak2_x, peak2_y = mu_2, norm.pdf(mu_2, mu_2, sigma)
# Define a common color for arrow and text
arrow_color = 'red'
plt.annotate('', xy=(peak2_x, peak2_y), xytext=(peak1_x, peak1_y),
             arrowprops=dict(facecolor=arrow_color, arrowstyle='->', linewidth=2))

# Add label below the arrow
plt.text((peak1_x + peak2_x)/2, min(peak1_y, peak2_y) - 0.1,
         'Projection', color='black', fontweight = 'bold', fontsize=14, ha='center')
         
# Final plot tweaks
plt.title("1D Gaussian Projection with Mean Constraint Only (Variance Fixed)")
plt.xlabel("Value of $y$")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()