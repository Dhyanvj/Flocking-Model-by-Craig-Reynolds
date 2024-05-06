# Task 1
import numpy as np
import matplotlib.pyplot as plt

class Boid:
    def __init__(self, x, y, vx, vy, inner_radius, outer_radius):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

def separation(boid, nearby_boids):
    """Separation behavior."""
    separation_vector = np.zeros(2)
    for other_boid in nearby_boids:
        distance = np.linalg.norm(boid.position - other_boid.position)
        if 0 < distance < boid.inner_radius:
            separation_vector += (boid.position - other_boid.position) / distance
    return separation_vector

def cohesion(boid, nearby_boids):
    """Cohesion behavior."""
    if not nearby_boids:
        return np.zeros(2)
    center_of_mass = np.mean([other_boid.position for other_boid in nearby_boids], axis=0)
    cohesion_vector = (center_of_mass - boid.position) / np.linalg.norm(center_of_mass - boid.position)
    return cohesion_vector

def alignment(boid, nearby_boids):
    """Alignment behavior."""
    if not nearby_boids:
        return np.zeros(2)
    average_velocity = np.mean([other_boid.velocity for other_boid in nearby_boids], axis=0)
    alignment_vector = (average_velocity - boid.velocity) / np.linalg.norm(average_velocity - boid.velocity)
    return alignment_vector

def update_boid(boid, all_boids):
    """Update boid's position and velocity based on behaviors."""
    separation_vector = separation(boid, all_boids)
    cohesion_vector = cohesion(boid, all_boids)
    alignment_vector = alignment(boid, all_boids)

    # Adjust weights based on your preference
    separation_weight = 1.5
    cohesion_weight = 1.0
    alignment_weight = 1.0

    # Update velocity
    boid.velocity += separation_weight * separation_vector + cohesion_weight * cohesion_vector + alignment_weight * alignment_vector
    # Limit the velocity to a maximum value if needed
    max_velocity = 5.0
    speed = np.linalg.norm(boid.velocity)
    if speed > max_velocity:
        boid.velocity = max_velocity * boid.velocity / speed

    # Update position
    boid.position += boid.velocity

# Example usage:
num_boids = 10
boids = [Boid(np.random.rand()*10, np.random.rand()*10, np.random.rand()*2-1, np.random.rand()*2-1, 2, 5) for _ in range(num_boids)]

# Simulation steps
for _ in range(100):
    for boid in boids:
        nearby_boids = [other_boid for other_boid in boids if np.linalg.norm(boid.position - other_boid.position) < boid.outer_radius and boid != other_boid]
        update_boid(boid, nearby_boids)

# Visualization (scatter plot)
x_values = [boid.position[0] for boid in boids]
y_values = [boid.position[1] for boid in boids]
plt.scatter(x_values, y_values)
plt.show()
