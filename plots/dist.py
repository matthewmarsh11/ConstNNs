import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Define distributions ===
# Original (from model)
mu_1 = 2.0
sigma_1 = 0.5

# Projected (should satisfy constraint zone)
mu_2 = 0.0
sigma_2 = 0.2

# Constraint center and tolerance (soft equality)
c = 0.0          # target value for equality
epsilon = 0.5    # tolerance around c
delta = 0.05     # 95% confidence ⇒ 1 - delta

# Range for plotting
x = np.linspace(-1, 3.5, 1000)
pdf_1 = norm.pdf(x, mu_1, sigma_1)
pdf_2 = norm.pdf(x, mu_2, sigma_2)

# Probability mass inside constraint zone
p1 = norm.cdf(c + epsilon, mu_1, sigma_1) - norm.cdf(c - epsilon, mu_1, sigma_1)
p2 = norm.cdf(c + epsilon, mu_2, sigma_2) - norm.cdf(c - epsilon, mu_2, sigma_2)

# === Plot ===
plt.figure(figsize=(10, 5))

# Plot both distributions
plt.plot(x, pdf_1, label=f'Unconstrained Network Output, P(y) on constraint', linewidth=2)
plt.plot(x, pdf_2, label=f'Projected Network Output, Q(y) on constraint', linewidth=2, linestyle='--')

# Shade the constraint zone
plt.axvspan(c - epsilon, c + epsilon, color='gray', alpha=0.3, label='Constraint Zone')

# Annotate projection
y1 = norm.pdf(mu_1, mu_1, sigma_1)
y2 = norm.pdf(mu_1, mu_2, sigma_2)
# Calculate the peaks of the distributions
peak1_x, peak1_y = mu_1, norm.pdf(mu_1, mu_1, sigma_1)
peak2_x, peak2_y = mu_2, norm.pdf(mu_2, mu_2, sigma_2)

# Draw arrow from peak of first distribution to peak of second
plt.annotate('', xy=(peak2_x, peak2_y), xytext=(peak1_x, peak1_y),
             arrowprops=dict(facecolor='red', arrowstyle='->'))

# Add text label along the arrow path
plt.text((peak1_x + peak2_x)/2 - 0.1, (peak1_y + peak2_y)/2 + 0.1, 
         'Projection', color='red', fontsize=10)

# Add epsilon arrows spanning the constraint zone
arrow_y_pos = max(pdf_1.max(), pdf_2.max()) * 0.15  # Position arrows at 15% of max height
# Left arrow (from -epsilon to center)
plt.annotate('', xy=(c, arrow_y_pos), xytext=(c - epsilon, arrow_y_pos),
             arrowprops=dict(facecolor='black', arrowstyle='<->'))
plt.text(c - epsilon/2, arrow_y_pos * 1.3, r'$\epsilon/2$', fontsize=12, ha='center')

# Right arrow (from center to +epsilon)
plt.annotate('', xy=(c + epsilon, arrow_y_pos), xytext=(c, arrow_y_pos),
             arrowprops=dict(facecolor='black', arrowstyle='<->'))
plt.text(c + epsilon/2, arrow_y_pos * 1.3, r'$\epsilon/2$', fontsize=12, ha='center')

# Final plot tweaks
plt.title("1D Gaussian Projection with Probabilistic Equality Constraint (±ε Zone)")
plt.xlabel("Residual of Constraint")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()