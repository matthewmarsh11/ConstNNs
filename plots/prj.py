import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

def draw_confidence_ellipse(mean, cov, ax, confidence=0.95, facecolor='none', edgecolor='gray', label=None, **kwargs):
    # Confidence scaling factor from chi-squared distribution with 2 DOF
    from scipy.stats import chi2
    s = np.sqrt(chi2.ppf(confidence, df=2))

    # Compute eigenvalues and eigenvectors of covariance
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * s * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=facecolor, edgecolor=edgecolor, label=label, **kwargs)
    ax.add_patch(ellipse)

# === Define distributions ===
mu1 = np.array([1.5, 1.5])
cov1 = np.array([[0.5, 0.03], [0.01, 0.7]])

mu2 = np.array([0.0, 0.0])  # Projected to constraint zone
cov2 = np.array([[0.1, 0.03], [0.03, 0.05]])

# Constraint target (soft equality center)
c = np.array([0.0, 0.0])
epsilon = 0.35  # Tolerance around the mean in Mahalanobis distance

# Create grid
x, y = np.mgrid[-2:4:0.05, -2:4:0.05]
pos = np.dstack((x, y))

rv1 = multivariate_normal(mu1, cov1)
rv2 = multivariate_normal(mu2, cov2)

z1 = rv1.pdf(pos)
z2 = rv2.pdf(pos)

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 6))

# Contours
ax.contour(x, y, z1, levels=5, cmap='Blues')
ax.contour(x, y, z2, levels=5, cmap='Oranges', linestyles='dashed')

# Means
ax.plot(*mu1, 'bo', label='Original Mean')
ax.plot(*mu2, 'ro', label='Projected Mean')

# Projection arrow
ax.annotate('', xy=mu2, xytext=mu1,
            arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
            annotation_clip=False)
ax.text(*(mu1 + mu2)/2 + np.array([0.8, 0.6]), 'Projection', color='black', fontweight='bold', fontsize=14)

# Constraint ellipse (soft equality constraint zone)
constraint_cov = np.array([
    [10000000, 0.0],
    [0.0, 0.2]
])

# constraint_cov = np.eye(2,2) * (epsilon**2)
draw_confidence_ellipse(c, constraint_cov, ax,
                        facecolor='gray', alpha=0.2, edgecolor='black',
                        label=f'Isotropic Constraint Zone')

# Final plot
ax.set_title("2D Gaussian Projection with Soft Equality Constraint (ε Zone)")
ax.set_xlabel("Residual of Constraint 1")
ax.set_ylabel("Residual of Constraint 2")
ax.legend()
ax.set_xlim(-1.2, 3.5)
ax.set_ylim(-1.2, 3.5)
ax.set_aspect('equal')
ax.grid(True)
plt.tight_layout()
plt.show()