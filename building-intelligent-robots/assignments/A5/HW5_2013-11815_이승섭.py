# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Map class
class Map2D:
    '''
    Generates a 2D map of specified size, goal position, initial robot position and obstacles for localization.
    '''
    def __init__(self, space_size, goal_pos, num_obstacles):
        '''
        Initialize with size and number of obstacles.
        '''
        self.space_size = space_size
        self.goal_pos = goal_pos
        self.GT_pos = np.random.uniform(0, space_size[0], 2)
        self.obstacles = self.gen_obstacles(num_obstacles)

    def gen_obstacles(self, num_obstacles):
        '''
        Generates random obstacles in the 2D space.
        '''
        obstacles = np.random.uniform(0, self.space_size[0], (num_obstacles, 2)).astype(float)
        return obstacles
    
    def get_GT(self):
        '''
        Returns the true position of the robot.
        '''
        return self.GT_pos
    
    def get_goal(self):
        '''
        Returns the goal position of the robot.
        '''
        return self.goal_pos
    
    def update_GT(self, motion, noise = 0.0):
        '''
        Updates the true position of the robot with given motion.
        '''
        self.GT_pos += motion + np.random.normal(0, noise, self.GT_pos.shape)
        self.GT_pos = np.clip(self.GT_pos, 0, self.space_size[0] - 1)


# ParticleFilter class
class ParticleFilter:
    '''
    Implements a naive Particle filter for 2D localization. 
    '''
    def __init__(self, num_particles, map2D, sensor_noise, motion_noise):
        '''
        Initialize with number of particles, map, sensor noise, and motion noise. Weights are uniformly initialized.
        '''
        self.num_particles = num_particles
        self.sensor_noise = sensor_noise
        self.motion_noise = motion_noise
        self.map = map2D
        self.particles = self.init_particles()
        self.weights = np.ones(num_particles) / num_particles

    def init_particles(self):
        '''
        Returns particles randomly initialized in the 2D space.
        '''
        particles = np.array([
            np.random.uniform(0, self.map.space_size[0], self.num_particles),
            np.random.uniform(0, self.map.space_size[1], self.num_particles)
        ]).T.astype(float)
        return particles

    def move(self, motion):
        '''
        Void function that moves particles with given motion and noise. Particles are clipped to stay within bounds.
        '''
        self.particles += motion + np.random.normal(0, self.motion_noise, self.particles.shape)
        self.particles = np.clip(self.particles, 0, self.map.space_size[0] - 1)

    def measure_dist(self, particle):
        '''
        Returns the distance of a particle to all obstacles in the map.
        '''
        try:
            distances = np.array([np.linalg.norm(particle - obstacle ) for obstacle in self.map.obstacles])
        except:
            pass # No obstacles
        return distances

    def update_weights(self, true_sensor_dists):
        '''
        Void function to update weights based on distance measurements. The measurements to each obstacle are naively assumed to be independent. Weights are normalized and a small value is added to avoid zero weights.
        '''
        for i, particle in enumerate(self.particles):
            particle_dists = self.measure_dist(particle)
            likelihoods = norm.pdf(true_sensor_dists, particle_dists, self.sensor_noise)
            self.weights[i] = np.prod(likelihoods) # Naive assumption of independence
        self.weights += 1e-300 # Small value to avoid zero weights.
        self.weights /= np.sum(self.weights)  # Normalize weights.

    def resample(self):
        '''
        Void function to resample particles based on their weights.
        '''
        indices = np.random.choice(len(self.particles), size=len(self.particles), p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pos(self):
        '''
        Estimates the position based on the weighted average of particle locations.
        '''
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def check_pos_var(self, var_threshold = 0.5):
        '''
        Checks if the particles have converged to the true position based on the variance of their locations.
        '''
        x_var = np.var(self.particles[:, 0])
        y_var = np.var(self.particles[:, 1])
        if (x_var > var_threshold or y_var > var_threshold):
            print(f'Variance in x: {x_var}, Variance in y: {y_var}')
        return x_var <= var_threshold and y_var <= var_threshold

# Robot class
class Robot:
    def __init__(self, map2D, pf, var_threshold):
        self.map2D = map2D
        self.pf = pf
        self.var_threshold = var_threshold
        self.trajectory = [self.map2D.get_GT()]

    def avoid_obstacles(self, motion):
        for obstacle in self.map2D.obstacles:
            dist_to_obstacle = np.linalg.norm(self.map2D.get_GT() - obstacle)
            if dist_to_obstacle < 4.0:
                obstacle_dir = self.map2D.get_GT() - obstacle
                obstacle_dir /= np.linalg.norm(obstacle_dir)
                motion += 0.2 * obstacle_dir
        return motion / np.linalg.norm(motion)

    def move(self):
        if not self.pf.check_pos_var(var_threshold=self.var_threshold):
            motion = np.random.uniform(-1, 1, 2)
        else:
            direction = self.map2D.get_goal() - self.map2D.get_GT()
            motion = 0.5 * direction / np.linalg.norm(direction)
            motion = self.avoid_obstacles(motion)

        self.map2D.update_GT(motion)
        self.trajectory.append(self.map2D.get_GT())
        self.pf.move(motion)

        true_sensor_dists = self.pf.measure_dist(self.map2D.get_GT())
        self.pf.update_weights(true_sensor_dists)
        self.pf.resample()

        return self.pf.estimate_pos()


# ShowMap Class for visualization
class ShowMap:
    def __init__(self, robot, pf, space_size):
        self.robot = robot
        self.pf = pf
        self.space_size = space_size

    def update(self, i):
        est_pos = self.robot.move()
        if np.linalg.norm(est_pos - self.robot.map2D.get_goal()) < 1.0:
            self.ani.event_source.stop()
        return self.plot(i, est_pos)

    def plot(self, i, est_pos):
        plt.clf()
        plt.xlim(0, self.space_size[0])
        plt.ylim(0, self.space_size[1])

        for obs in self.robot.map2D.obstacles:
            square = plt.Rectangle(obs, 2, 2, color='grey')
            plt.gca().add_patch(square)

        plt.scatter(self.robot.map2D.get_GT()[0], self.robot.map2D.get_GT()[1], c='blue', s=100, label="True Position")
        plt.scatter(self.robot.map2D.get_goal()[0], self.robot.map2D.get_goal()[1], c='green', s=100, label="Goal Position")
        particle_sizes = self.pf.weights * 5000
        plt.scatter(self.pf.particles[:, 0], self.pf.particles[:, 1], color='red', alpha=0.4, s=particle_sizes, label="Particles (weighted)")
        plt.scatter(est_pos[0], est_pos[1], c='yellow', s=100, zorder=5, label="Estimated Position")
        plt.plot([pos[0] for pos in self.robot.trajectory], [pos[1] for pos in self.robot.trajectory], c='blue', label="Trajectory")

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f'Frame {i}')
        return plt

    def animate(self):
        fig = plt.figure()
        self.ani = FuncAnimation(fig, self.update, frames=range(100), repeat=False)
        plt.show()


# Main function
if __name__ == '__main__':
    # Get parameters
    space_size = (100, 100)
    goal_pos = np.array([90, 90])
    num_obstacles = 100
    num_particles = 150
    sensor_noise = 1
    motion_noise = 0.5
    var_threshold = 1.5

    # Initialize classes
    map2D = Map2D(space_size, goal_pos, num_obstacles)
    pf = ParticleFilter(num_particles, map2D, sensor_noise, motion_noise)
    robot = Robot(map2D, pf, var_threshold)
    show_map = ShowMap(robot, pf, space_size)

    # Run animation
    show_map.animate()