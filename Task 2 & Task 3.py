# Task 2
import numpy as np
import pandas as pd

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


def save_to_csv(boids, filename):
    data = {"X_Position": [boid.position[0] for boid in boids],
            "Y_Position": [boid.position[1] for boid in boids],
            "X_Velocity": [boid.velocity[0] for boid in boids],
            "Y_Velocity": [boid.velocity[1] for boid in boids]}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def run_simulation(num_boids, num_steps, filename):
    # Using Uniform Distribution For Seeding
    boids = [Boid(np.random.rand()*10, np.random.rand()*10, np.random.rand()*2-1, np.random.rand()*2-1, 2, 5) for _ in range(num_boids)]

    positions = []
    velocities = []

    for step in range(num_steps):
        current_positions = [(boid.position[0], boid.position[1]) for boid in boids]
        current_velocities = [(boid.velocity[0], boid.velocity[1]) for boid in boids]

        positions.append(current_positions)
        velocities.append(current_velocities)

        for boid in boids:
            nearby_boids = [other_boid for other_boid in boids if np.linalg.norm(boid.position - other_boid.position) < boid.outer_radius and boid != other_boid]
            update_boid(boid, nearby_boids)

    save_to_csv(boids, filename)

# Run simulations
run_simulation(10, 200, "simulation_10_boids.csv")
run_simulation(100, 200, "simulation_100_boids.csv")


# Task 3
import itertools

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


def save_to_csv(boids, filename):
    data = {"X_Position": [boid.position[0] for boid in boids],
            "Y_Position": [boid.position[1] for boid in boids],
            "X_Velocity": [boid.velocity[0] for boid in boids],
            "Y_Velocity": [boid.velocity[1] for boid in boids]}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def run_simulation_with_behaviors(behaviors, num_steps, filename):
    boids = [Boid(np.random.rand()*10, np.random.rand()*10, np.random.rand()*2-1, np.random.rand()*2-1, 2, 5) for _ in range(100)]

    positions = []
    velocities = []

    for step in range(num_steps):
        current_positions = [(boid.position[0], boid.position[1]) for boid in boids]
        current_velocities = [(boid.velocity[0], boid.velocity[1]) for boid in boids]

        positions.append(current_positions)
        velocities.append(current_velocities)

        for boid in boids:
            nearby_boids = [other_boid for other_boid in boids if np.linalg.norm(boid.position - other_boid.position) < boid.outer_radius and boid != other_boid]

            if "Separation" in behaviors and "Cohesion" in behaviors:
                update_boid(boid, nearby_boids)
            elif "Separation" in behaviors and "Alignment" in behaviors:
                update_boid(boid, nearby_boids)
            elif "Cohesion" in behaviors and "Alignment" in behaviors:
                update_boid(boid, nearby_boids)

    save_to_csv(boids, filename)

# Run simulations for each combination of behaviors
combinations = ["Separation&Cohesion", "Separation&Alignment", "Cohesion&Alignment"]

for combination in combinations:
    behaviors = combination.split("&")
    run_simulation_with_behaviors(behaviors, 200, f"simulation_{combination}_100_boids.csv")
