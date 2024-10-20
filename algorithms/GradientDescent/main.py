import numpy as np
import matplotlib.pyplot as plt

"""
# fundtion of x
def y_function(x):
    return x ** 2

# derivative of function of x
def y_derivative(x):
    return 2 * x

# defining doman of y
x = np.arange(-100, 100, 0.1)
y = y_function(x)

# defining a random point on the function of x
x0 = 50
current_pos = (x0, y_function(x0))
"""

# fundtion of x
def y_function(x):
    return np.sin(x)

# derivative of function of x
def y_derivative(x):
    return np.cos(x)

# defining doman of y
x = np.arange(-5, 5, 0.1)
y = y_function(x)

# defining a random point(current loss) on the function of x
x0 = 1.2
current_pos = (x0, y_function(x0))

# defining learning_rate and adjustment
learning_rate = 0.01

for _ in range(1000):
    new_x = current_pos[0] - learning_rate * y_derivative(current_pos[0])
    new_y = y_function(new_x)
    current_pos = (new_x, new_y)

    # plotting the scenario
    plt.plot(x, y)
    plt.scatter(current_pos[0], current_pos[1], c = 'red')
    plt.pause(learning_rate)
    plt.clf()