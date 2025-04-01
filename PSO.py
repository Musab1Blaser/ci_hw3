import math 
import random
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    if -2 < x < 2 and -1 < y < 3:
        return 100*(x**2 - y)**2 + (1 - x)**2
    else:
        print(f"Rosenbrock function is only defined for x in (-2, 2) and y in (-1, 3). Given x: {x}, y: {y}")
        return float('inf') 

def greiwank(x, y):
    if -30 < x < 30 and -30 < y < 30:
        return 1+ (x**2 + y**2)/4000 - math.cos(x)*math.cos(y/math.sqrt(2))
    else:
        print(f"Greiwank function is only defined for x and y in (-30, 30). Given x: {x}, y: {y}")
        return float('inf')

def init_population(pop_size, bounds):

    dim = len(bounds)
    positions = []
    velocities = []

    for _ in range(pop_size):
        position = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        velocity = [random.uniform(-5, 5) for _ in range(dim)]
        positions.append(position)
        velocities.append(velocity)

    return positions, velocities

def update(positions, velocities, p_best, g_best, w, c1, c2, bounds):

    dim = len(bounds)
    for i in range(len(positions)):
        for d in range(dim):
            positions[i][d] += velocities[i][d]
            
            if positions[i][d] < bounds[d][0]:
                positions[i][d] = bounds[d][0] + 0.001
            elif positions[i][d] > bounds[d][1]:
                positions[i][d] = bounds[d][1] - 0.001
            
            r1 = random.random()
            r2 = random.random()

            velocities[i][d] = w*velocities[i][d] + c1*r1*(p_best[i][d] - positions[i][d]) + c2*r2*(g_best[d] - positions[i][d])
    
            if velocities[i][d] < -1:
                velocities[i][d] = -1
            elif velocities[i][d] > 1:
                velocities[i][d] = 1

def fitness(positions, p_best, g_best, fitness_func):
    
    for i in range(len(positions)):
        fit = fitness_func(positions[i][0], positions[i][1])
        if fit != float('inf'):
            if fit < p_best[i][2]:
                p_best[i][0] = positions[i][0]
                p_best[i][1] = positions[i][1]
                p_best[i][2] = fit
                if fit < g_best[2]:
                    g_best[0] = positions[i][0]
                    g_best[1] = positions[i][1]
                    g_best[2] = fit
        else:
            print("Invalid fitness value")

def init_pso(pop_size, bounds, fitness_func):
    
    positions, velocities = init_population(pop_size, bounds)
    
    p_best = [[positions[i][0], positions[i][1], fitness_func(positions[i][0], positions[i][1])] for i in range(pop_size)]
    g_best = [positions[0][0], positions[0][1], fitness_func(positions[0][0], positions[0][1])]

    for i in range(pop_size):
        if p_best[i][2] < g_best[2]:
            g_best[0] = p_best[i][0]
            g_best[1] = p_best[i][1]
            g_best[2] = p_best[i][2]

    return positions, velocities, p_best, g_best

func = 1 # 0 for Rosenbrock, 1 for Greiwank
POPULATION_SIZE = 30
if func == 0:
    BOUNDS = [(-2, 2), (-1, 3)]  # Rosenbrock function bounds
    positions, velocities, p_best, g_best = init_pso(POPULATION_SIZE, BOUNDS, rosenbrock)
else:
    BOUNDS = [(-30, 30), (-30, 30)]  # Greiwank function bounds
    positions, velocities, p_best, g_best = init_pso(POPULATION_SIZE, BOUNDS, greiwank)

MAX_ITERATIONS = 250
W = 0.5
C1 = 1.5
C2 = 1.5

iters = 0
bests = []
averages = []

while iters < MAX_ITERATIONS:
    update(positions, velocities, p_best, g_best, W, C1, C2, BOUNDS)

    if func == 0:
        fitness(positions, p_best, g_best, rosenbrock)
    else:
        fitness(positions, p_best, g_best, greiwank)

    iters += 1
    bests.append(g_best.copy())
    averages.append(sum(p[2] for p in p_best) / len(p_best))
    print(f"Iteration: {iters} | Best position: {bests[-1][0]}, {bests[-1][1]} | Best fitness: {bests[-1][2]} | Average fitness: {averages[-1]}")

bests_fitness = [b[2] for b in bests]
iterations = list(range(1, len(bests_fitness) + 1))
plt.plot(iterations, bests_fitness, label='Best Fitness')
plt.plot(iterations, averages, label='Average Fitness')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.title('Fitness vs Iterations')
plt.legend()
plt.grid()
plt.show()