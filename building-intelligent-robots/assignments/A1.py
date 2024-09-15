import numpy as np
import matplotlib.pyplot as plt

# Function to simulate Brownian motion
def brownian_motion_1d(n_steps, step_size=1):
    steps = np.random.normal(0, step_size, size=n_steps)
    return np.cumsum(steps)

def brownian_motion_2d(n_steps, step_size=1):
    steps_x = np.random.normal(0, step_size, size=n_steps)
    steps_y = np.random.normal(0, step_size, size=n_steps)
    x = np.cumsum(steps_x)
    y = np.cumsum(steps_y)
    return x, y

def brownian_motion_3d(n_steps, step_size=1):
    steps_x = np.random.normal(0, step_size, size=n_steps)
    steps_y = np.random.normal(0, step_size, size=n_steps)
    steps_z = np.random.normal(0, step_size, size=n_steps)
    x = np.cumsum(steps_x)
    y = np.cumsum(steps_y)
    z = np.cumsum(steps_z)
    return x, y, z

# Function to plot and save 1D Brownian motion
def plot_brownian_1d(n_steps, step_size=1, file_name='brownian_1d.png'):
    x = np.arange(n_steps)
    y = brownian_motion_1d(n_steps, step_size)
    plt.figure()
    plt.plot(x, y)
    plt.title('1D Brownian Motion')
    plt.xlabel('Steps')
    plt.ylabel('Position')
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()

# Function to plot and save 2D Brownian motion
def plot_brownian_2d(n_steps, step_size=1, file_name='brownian_2d.png'):
    x, y = brownian_motion_2d(n_steps, step_size)
    plt.figure()
    plt.plot(x, y)
    plt.title('2D Brownian Motion')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(file_name)
    plt.close()

# Function to plot and save 3D Brownian motion
def plot_brownian_3d(n_steps, step_size=1, file_name='brownian_3d.png'):
    from mpl_toolkits.mplot3d import Axes3D
    x, y, z = brownian_motion_3d(n_steps, step_size)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_title('3D Brownian Motion')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    plt.savefig(file_name)
    plt.close()

# Parameters
n_steps = 1000  # Number of steps
step_size = 1   # Size of each step

# Generate and save plots
plot_brownian_1d(n_steps, step_size, file_name='./brownian_1d.png')
plot_brownian_2d(n_steps, step_size, file_name='./brownian_2d.png')
plot_brownian_3d(n_steps, step_size, file_name='./brownian_3d.png')

print("Brownian motion plots have been saved.")