import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def generate_kernel_def1(x, y, λ, θ, ϕ, σ, γ):
    x_prime = x * np.cos(θ) + y * np.sin(θ)
    y_prime = -x * np.sin(θ) + y * np.cos(θ)
    
    real_part = np.exp(- (x_prime**2 + γ**2 * y_prime**2) / (2 * σ**2)) * np.cos(2 * np.pi / λ * x_prime + ϕ)
    return real_part

def generate_kernel_def2(x, y, f_0, θ, γ, η):
    x_prime = x * np.cos(θ) + y * np.sin(θ)
    y_prime = -x * np.sin(θ) + y * np.cos(θ)
    
    norm_factor = (f_0**2) / (np.pi * γ * η)
    real_part = norm_factor * np.exp(- (f_0**2 / γ**2) * (x_prime**2 + y_prime**2)) * np.cos(2 * np.pi * f_0 * x_prime)
    return real_part

# Parameters
size = 100  # Increased size for better resolution
x_range = np.linspace(-3, 3, size)
y_range = np.linspace(-3, 3, size)
x, y = np.meshgrid(x_range, y_range)

# Create a grid for frequency values
frequency_range = np.linspace(0.1, 0.5, 100)  # Frequency values from 0.1 to 0.5 for more variation

# Parameters for both definitions
θ = 0  # Orientation angle
ϕ = 0  # Phase offset
σ = 2.8  # Standard deviation for definition 1
γ = 0.3  # Aspect ratio for both definitions
η = 1.0  # Parameter for definition 2

# Define f_0 as half of the maximum frequency in the range
f_0_max = max(frequency_range)
f_0_range = f_0_max / 2  # f_0 is half of the maximum frequency

# Initialize plots
fig = plt.figure(figsize=(14, 7))

# Plot for Definition 1
ax1 = fig.add_subplot(121, projection='3d')
X, Y, Freq = np.meshgrid(x_range, y_range, frequency_range, indexing='ij')
Real_part_def1 = np.zeros_like(X)

for i, f in enumerate(frequency_range):
    λ = 1 / f  # Calculate λ from frequency
    Real_part_def1[:, :, i] = generate_kernel_def1(X[:, :, i], Y[:, :, i], λ, θ, ϕ, σ, γ)

# Plot the first slice of frequency
surf1 = ax1.plot_surface(X[:, :, 0], Y[:, :, 0], Real_part_def1[:, :, 0], 
                         facecolors=plt.cm.viridis(Normalize(vmin=np.min(Real_part_def1), vmax=np.max(Real_part_def1))(Real_part_def1[:, :, 0])),
                         rstride=1, cstride=1, antialiased=True)
ax1.set_title('Gabor Filter Definition 1 - Real Part')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('z = f')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([np.min(Real_part_def1), np.max(Real_part_def1)])
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot for Definition 2
ax2 = fig.add_subplot(122, projection='3d')
X, Y = np.meshgrid(x_range, y_range, indexing='ij')
Real_part_def2 = np.zeros_like(X)

# Calculate Real_part_def2 using f_0 as half of the maximum frequency
Real_part_def2 = generate_kernel_def2(X, Y, f_0_range, θ, γ, η)

# Normalize for better visualization
norm_def2 = Normalize(vmin=np.min(Real_part_def2), vmax=np.max(Real_part_def2))
# Plot the result
surf2 = ax2.plot_surface(X, Y, Real_part_def2, 
                         facecolors=plt.cm.viridis(norm_def2(Real_part_def2)),
                         rstride=1, cstride=1, antialiased=True)
ax2.set_title('Gabor Filter Definition 2 - Real Part')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('z = f')
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([np.min(Real_part_def2), np.max(Real_part_def2)])
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()


