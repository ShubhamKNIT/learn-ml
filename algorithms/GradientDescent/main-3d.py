import numpy as np
import matplotlib.pyplot as plt

def z_function(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5.0

def z_gradient(x, y):
    # Correctly calculate the gradients for z_function
    dz_dx = (5 * np.cos(5 * x) * np.cos(5 * y))  # Derivative with respect to x
    dz_dy = (-5 * np.sin(5 * x) * np.sin(5 * y))  # Derivative with respect to y
    return dz_dx, dz_dy

# Generate the grid for the surface
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

# Define initial position (current loss)
x0, y0 = 0.2, 0.3
current_pos = (x0, y0, z_function(x0, y0))

learning_rate = 0.01

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='plasma', zorder=0)
ax.set_title('Gradient Descent Visualization')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

for _ in range(100):
    X_derivative, Y_derivative = z_gradient(current_pos[0], current_pos[1])
    
    # Update position
    X_new = current_pos[0] - learning_rate * X_derivative
    Y_new = current_pos[1] - learning_rate * Y_derivative
    current_pos = (X_new, Y_new, z_function(X_new, Y_new))

    # Clear and redraw the surface and the current position
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='plasma', zorder=0)
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], c='green', s=100, zorder=1)

    plt.pause(0.1)  # Pause to visualize the update

plt.show()
