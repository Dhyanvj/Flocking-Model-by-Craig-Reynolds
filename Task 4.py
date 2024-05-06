# Task 4
# Modification 1: Introducing Obstacles

class Obstacle:
    def __init__(self, x, y, radius):
        self.position = np.array([x, y])
        self.radius = radius

def avoid_obstacles(boid, obstacles):
    """Avoidance behavior for obstacles."""
    avoidance_vector = np.zeros(2)
    for obstacle in obstacles:
        distance = np.linalg.norm(boid.position - obstacle.position)
        if distance < boid.outer_radius + obstacle.radius:
            avoidance_vector += (boid.position - obstacle.position) / distance
    return avoidance_vector

def update_boid_with_obstacles(boid, all_boids, obstacles):
    """Update boid's position and velocity with obstacle avoidance."""
    separation_vector = separation(boid, all_boids)
    cohesion_vector = cohesion(boid, all_boids)
    alignment_vector = alignment(boid, all_boids)
    avoidance_vector = avoid_obstacles(boid, obstacles)

    # Adjust weights based on your preference
    separation_weight = 1.5
    cohesion_weight = 1.0
    alignment_weight = 1.0
    avoidance_weight = 2.0

    # Update velocity
    boid.velocity += (separation_weight * separation_vector +
                      cohesion_weight * cohesion_vector +
                      alignment_weight * alignment_vector +
                      avoidance_weight * avoidance_vector)

    # Limit the velocity to a maximum value if needed
    max_velocity = 5.0
    speed = np.linalg.norm(boid.velocity)
    if speed > max_velocity:
        boid.velocity = max_velocity * boid.velocity / speed

    # Update position
    boid.position += boid.velocity

# Modification 2: Random Perturbations

def introduce_perturbations(boid, perturbation_scale=0.1):
    """Introduce random perturbations to boid's velocity."""
    perturbation = perturbation_scale * np.random.rand(2)
    return perturbation

def update_boid_with_perturbations(boid):
    """Update boid's position and velocity with random perturbations."""
    perturbation = introduce_perturbations(boid)
    boid.velocity += perturbation

    # Limit the velocity to a maximum value if needed
    max_velocity = 5.0
    speed = np.linalg.norm(boid.velocity)
    if speed > max_velocity:
        boid.velocity = max_velocity * boid.velocity / speed

    # Update position
    boid.position += boid.velocity

# Simulation Runs

# Simulation with Obstacles
obstacles = [Obstacle(3, 5, 1), Obstacle(8, 8, 1)]
for obstacle in obstacles:
    plt.scatter(obstacle.position[0], obstacle.position[1], color='red', marker='x')

for _ in range(200):
    for boid in boids:
        nearby_boids = [other_boid for other_boid in boids if np.linalg.norm(boid.position - other_boid.position) < boid.outer_radius and boid != other_boid]
        update_boid_with_obstacles(boid, nearby_boids, obstacles)

save_to_csv(boids, "simulation_with_obstacles.csv")
plt.show()

# Simulation with Perturbations
for _ in range(200):
    for boid in boids:
        update_boid_with_perturbations(boid)

save_to_csv(boids, "simulation_with_perturbations.csv")
plt.scatter([boid.position[0] for boid in boids], [boid.position[1] for boid in boids])
plt.show()
