import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.patches as patches

from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD

df = pd.read_csv("vrp7.txt", delimiter=r"\s+", header=None)  # "\s+" handles multiple spaces/tabs
df.columns = ["id", "x", "y", "demand"]  # Rename columns if needed

positions = list(zip(df['x'],df['y']))
demand = df['demand']

position_scores = np.random.rand(len(positions))


#For testing purposes
# positions = positions[:5]
# position_scores = position_scores[:5]
# demand = demand[:5]

print("Total number of cities is: ", len(positions))
# print(demand)
# print(position_scores)

## Find minimum distance you could travel
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def TSP_minDistance(positions):
    locations = positions[:]
    path = [locations.pop(0)]
    total_dist = 0

    while locations:
        nearest = min(locations, key=lambda d: euclidean_distance(path[-1], d))
        total_dist += euclidean_distance(path[-1], nearest)
        path.append(nearest)
        locations.remove(nearest)

    return path, total_dist

_, min_dist = TSP_minDistance(positions)


factored_demand = sorted(demand, reverse=True)
factored_demand = [val * (1 / (i + 1)) for i, val in enumerate(factored_demand)]
print(len(factored_demand))

factored_demand = np.sum(factored_demand)
print(factored_demand)

### Problem Definition
class PTSP:
    def __init__(self, D, min_dist, factored_demand):
        self.D = D # Number of particles
        self.min_dist = min_dist
        self.factored_demand = factored_demand

    def evaluate(self,x, coords, demand):
        # Calculate for distance first
        position_scores = x
        sorted_positions = np.argsort(position_scores) #Arranges in ascending order
        rankings = np.empty_like(sorted_positions)
        rankings[sorted_positions] = np.arange(len(position_scores), 0, -1)  # Start with the location with the highest score
        # print(rankings)

        dist_travelled = 0
        accumulated_demand = 0
        # print(rankings)
        for i in range(len(rankings)):
            cur_pos = i+1
            cur_coords_idx = np.argwhere(rankings == cur_pos)[0][0]
            cur_coords = np.array(coords[cur_coords_idx])

            #Calculate demand
            cur_demand = 1/cur_pos * demand[cur_coords_idx]
            accumulated_demand += cur_demand
            # print(cur_demand)


            if cur_pos < len(rankings):
                next_pos = cur_pos+1
                next_coords_idx = np.argwhere(rankings == next_pos)[0][0]
                next_coords = np.array(coords[next_coords_idx])

                dist = np.linalg.norm(cur_coords - next_coords)
                dist_travelled += dist
        # print(accumulated_demand)

        # Distance Fitness score
        dist_score = round(self.min_dist / dist_travelled,6) # By applying mind dist we can normalize the distance travelled to see how close we are to approx perfection
        priority_score = round(accumulated_demand/self.factored_demand,6) # results are rounded otherwise floating point error occurs

        # Priority score calculations
        return np.array([dist_score, priority_score])

## Dominates function
def dominates(u, v):
    return (u>=v).all() and (u>v).any()

## Pareto Archive Definition
class Archive:

    def __init__(self):
        self.objective_vectors = []
        self.history = [] #History of objectives
        self.solution_vectors = []


    def update(self, x, y):
        to_remove = []

        for i in range(len(self.objective_vectors)):

            if dominates(self.objective_vectors[i], y):
                return
            elif dominates(y, self.objective_vectors[i]):
                to_remove.append(i)

        self.objective_vectors.append(y)
        self.solution_vectors.append(x)

        to_remove = sorted(to_remove, reverse=True)

        for i in to_remove:
            self.objective_vectors.pop(i)
            self.solution_vectors.pop(i)

    def save_history(self):
        self.history.append(self.objective_vectors.copy())

## Mutation methods
class MultiCrossOver:
    def __init__(self, num_of_points = 3):
        self.num_of_points = num_of_points

    def mutate(self, particle1, particle2):
        if len(particle1) != len(particle2):
            raise ValueError("Particles are of different lengths")
        crossover_points = np.random.choice(len(particle1), self.num_of_points, replace=False)

        particle1_new = particle1.copy()
        particle2_new = particle2.copy()

        particle1_points = particle1[crossover_points]
        particle2_points = particle2[crossover_points]

        particle1_new[crossover_points] = particle2_points
        particle2_new[crossover_points] = particle1_points

        return particle1_new, particle2_new, crossover_points




class AdditiveGaussianMutation:

    def __init__(self, std=0.1):
        self.std = std


    def mutate(self, x):
    # Make a copy of the solution and select a random decision
    # variable.
        xp = x.copy()
        idx = np.random.randint(xp.shape[0])

    # Set the chosen decision variable to be out of bounds.
    # We will iterate until we get an in-bound decision variable.
        xp[idx] = np.inf

        while xp[idx] < 0 or xp[idx] > 1:
    # Add a random Gaussian value (scaled with the 'std'
    # parameter) to the decision variable chosen.
            z = np.random.randn() * self.std
            xp[idx] = x[idx] + z

    # Return the mutated solution.
        return xp, np.array([idx])

class EPSO:

    def __init__(self, mutation1, mutation2, w = 0.2, c1 = 0.2, c2=0.2, c3=0.2):
        self.mutation1 = mutation1
        self.mutation2 = mutation2
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

        self.p_gd_score = 0
        self.p_gd = np.array([])

        self.p_id_score = np.array([])
        self.p_id = np.array([])

        self.w1 = 0.5
        self.w2 = 0.5

    def calc_velocity(self, v_id, x_id, demand):

        # beta = demand / np.sum(demand)
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)

        v_new = self.w * v_id + r1* self.c1 * (self.p_id - x_id) + r2 * self.c2 * (self.p_gd - x_id) # + self.c3 * beta
        return np.array(v_new)

    def calc_new_dist(self, cur_dist, v_new):
        new_dist = cur_dist + v_new
        new_dist = new_dist / np.sum(new_dist) #Normalize so it doesnt go crazy

        return np.array(new_dist)

    def check_score(self, cur_positions, demand, archive):
        fitness = []
        for i in range(len(cur_positions)):
            new_y = problem.evaluate(cur_positions[i], positions, demand)
            fitness.append(new_y)

            dist_score, priority_score = new_y
            archive.update(cur_positions[i], new_y)
            overall_score = self.w1 * dist_score + self.w2 * priority_score

            if overall_score > self.p_gd_score:
                self.p_gd_score = overall_score
                self.p_gd = cur_positions[i]

            if overall_score > self.p_id_score[i]:
                self.p_id_score[i] = overall_score
                self.p_id = cur_positions[i]

        fitness = np.array(fitness)

        return archive, fitness

    def mutations(self, positions, cur_idx):

        cur_positions = positions[cur_idx]
        second_parent_idx = np.random.choice([i for i in range(len(positions)) if i != cur_idx])
        second_parent = positions[second_parent_idx]

        #Cross over
        xp,_,_ = self.mutation2.mutate(cur_positions, second_parent)

        #additive Gaussian
        xp, _ = self.mutation1.mutate(xp)


        return xp


    def optimise(self, problem, positions, demand, niter, num_of_particles = 1):
        # Initialise a random starting solution.
        cur_v = np.random.rand(num_of_particles, problem.D)
        x = np.random.rand(num_of_particles, problem.D)
        archive = Archive()

        #Initialize scores
        self.p_gd = np.zeros((num_of_particles, problem.D))
        self.p_id_score = np.zeros(num_of_particles)
        self.p_id = np.zeros((num_of_particles, problem.D))

        #Check the scores
        archive, y = self.check_score(x, demand, archive)

        for j in tqdm(range(niter)):
            fitness = []
            for i in range(num_of_particles):
                # print(f"Solving for particle {i+1}")
                xp = self.mutations(x, i)
                # print(cur_positions[i].shape)
                # print(perturbed_positions.shape)
                # print("Hello")

                yp = problem.evaluate(xp, positions, demand)
                dist_score, priority_score = yp
                archive.update(xp, yp)

                overall_score = self.w1 * dist_score + self.w2 * priority_score

                if overall_score > self.p_gd_score:
                    self.p_gd_score = overall_score
                    self.p_gd = xp

                if overall_score > self.p_id_score[i]:
                    self.p_id_score[i] = overall_score
                    self.p_id = xp
                # print("shape of cur v is : ", cur_v.shape)
                # print("shape of cur v is : ", cur_v[i].shape)

                cur_v[i] = self.calc_velocity(cur_v[i], xp, demand)
                # print(cur_v[i])
                # print("shape of cur pos is : ", perturbed_positions.shape)
                # print("shape of cur pos is : ", perturbed_positions[i].shape)

                xp = self.calc_new_dist(cur_v[i], xp)


                if not dominates(y[i], yp):
                    x[i] = xp
                    y[i] = yp
            archive.save_history()


        return archive

def crowding_distance(solutions):
    num_of_sols, num_of_obj = solutions.shape

    # Initialize crowding distance for each individual
    crowding_distances = np.zeros(num_of_sols)

    # Iterate over each objective to calculate the crowding distance
    for i in range(num_of_obj):
        # Sort the population based on the i-th objective
        indices_sorted = np.argsort(solutions[:, i])
        sol_sorted = solutions[indices_sorted, i]

        # Boundary points get an infinite distance
        crowding_distances[indices_sorted[0]] = np.inf
        crowding_distances[indices_sorted[-1]] = np.inf

        min_value = np.min(sol_sorted)
        max_value = np.max(sol_sorted)

        # Calculate crowding distance for interior points
        for j in range(1, num_of_sols - 1):
            next_value = sol_sorted[j + 1]
            prev_value = sol_sorted[j - 1]
            norm_distance = (next_value - prev_value) / (max_value - min_value)
            crowding_distances[indices_sorted[j]] += norm_distance

    return crowding_distances


# problem = PTSP(5)
# dist_score, priority_score = problem.evaluate(position_scores, positions, demand)
# positions = positions[:5]
N = 200
folder_to_keep = f"images/{N}_particles"
os.makedirs(folder_to_keep, exist_ok=True)

##Verify experiment functions appropriately

# Initialise solutions
ori_sols = np.random.rand(2, len(positions))

# Cross over
crossover = MultiCrossOver(3)
new_sol_1, new_sol_2, crossover_points = crossover.mutate(ori_sols[0],ori_sols[1])
# print(f"Cross over result: {crossovered_problem}")
# blank_row = np.full_like(ori_sols[0], np.nan)  # Create a NaN row
data = pd.DataFrame([ori_sols[0], ori_sols[1] ,new_sol_1, new_sol_2],
                    index=['Parent 1', 'Parent 2', 'Child 1', 'Child 2'])

data = data.T

fig, ax = plt.subplots(figsize=(9, 16))
sns.heatmap(data, cmap="coolwarm", fmt=".2f" ,annot=True, cbar=False, linewidths=1, linecolor='black')
separation_index = 2  # Between index 1 (Parent 2) and 2 (Child 1)
ax.vlines(x=separation_index, ymin=-0.5, ymax=len(ori_sols[0]), colors="white", linewidth=2)

for i in range(len(crossover_points)):
    loc = crossover_points[i]
    for row in range(4):  # Highlight both parents and children
        rect = patches.Rectangle((row, loc), 1, 1, linewidth=5, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

plt.title("Crossover Visualization with Box Highlighting")
plt.xlabel('Particles')
plt.ylabel('Cities Priority')
plt.savefig(f"{folder_to_keep}/CrossOver_Visualisation.png", dpi=300, bbox_inches="tight")  # Save as high-quality image
plt.show()

# Gaussian Mutation
gaussian = AdditiveGaussianMutation()
gaussianed_problem, mutated_point = gaussian.mutate(ori_sols[0])

data = pd.DataFrame([ori_sols[0], gaussianed_problem],
                    index=['Parent 1', 'Child 1'])
data = data.T

fig, ax = plt.subplots(figsize=(9, 16))
sns.heatmap(data, cmap="coolwarm", fmt=".2f" ,annot=True, cbar=False, linewidths=1, linecolor='black')
separation_index = 1  # Between index 1 (Parent 2) and 2 (Child 1)
ax.vlines(x=separation_index, ymin=-0.5, ymax=len(ori_sols[0]), colors="white", linewidth=2)

for i in range(len(mutated_point)):
    loc = mutated_point[i]
    for row in range(4):  # Highlight both parents and children
        rect = patches.Rectangle((row, loc), 1, 1, linewidth=5, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

plt.title("Mutation Visualization with Box Highlighting")
plt.xlabel('Particles')
plt.ylabel('Cities Priority')
plt.savefig(f"{folder_to_keep}/Mutation_Visualisation.png", dpi=300, bbox_inches="tight")  # Save as high-quality image
plt.show()
plt.close()


## End of verifications
problem = PTSP(len(positions), min_dist, factored_demand)

X = np.random.rand(N, len(positions))
Y = np.array([problem.evaluate(X[i], positions, demand) for i in range(N)])
plt.scatter(Y[:,0], Y[:,1])



archive = Archive()
for x, y in zip(X,Y):
 archive.update(x, y)

A = np.array(archive.objective_vectors)
# print(A)
plt.scatter(A[:,0], A[:,1])
# plt.show()

problem = PTSP(len(positions), min_dist, factored_demand)
optimiser = EPSO(AdditiveGaussianMutation(), MultiCrossOver())
archive = optimiser.optimise(problem, positions, demand, 2000, N)
Y = np.array([archive.objective_vectors[i] for i in range(len(archive.objective_vectors))])

plt.scatter(Y[:,0], Y[:,1])
plt.xlabel("Distance Metric")
plt.ylabel("Priority Metric")
plt.savefig(f"{folder_to_keep}/pareto_front_withinit.png", dpi=300, bbox_inches="tight")
plt.show()

plt.close()
plt.scatter(Y[:,0], Y[:,1])
plt.xlabel("Distance Metric")
plt.ylabel("Priority Metric")
plt.savefig(f"{folder_to_keep}/pareto_front_withoutinit.png", dpi=300, bbox_inches="tight")
plt.close()

# Compute hypervolume
#Pymoo asumes minimization problem so we flip the problem into negative zone and set reference to 0, 0
reference = np.array([0, 0])
hv = HV(ref_point=reference)
hypervolume = hv(-Y)
# hypervolume = 1-hypervolume

# Computer crowding Distance
crowding_dist = crowding_distance(Y)
finite_distances = crowding_dist[np.isfinite(crowding_dist)]
average_crowding_dist = np.mean(finite_distances)

#Compute Generational distance
print("Now finding generational distance")

gd_list = []
num_of_epochs = []
average_gd = []
for i in range(len(archive.history)-1):
    cur_par = archive.history[i]
    cur_par_array = np.array([cur_par[j] for j in range(len(cur_par))])
    future_par = archive.history[i+1]
    future_par_array = np.array([future_par[j] for j in range(len(future_par))])
    gd = GD(future_par_array)

    average_gd.append(gd(cur_par_array))
    if (i+1)%100 == 0:
        average_gd = np.mean(average_gd)
        gd_list.append(average_gd)
        num_of_epochs.append(i+1)
        average_gd = []

plt.plot(num_of_epochs, gd_list, linestyle='-', marker='o', color='blue')
plt.xlabel("Iteration")
plt.ylabel("Generational Distance")
plt.title("Average change in generational Distance")
plt.grid(True)
plt.savefig(f"{folder_to_keep}/GenerationalDistancePlot.png", dpi=300, bbox_inches="tight")  # Save as high-quality image
plt.show()

text_to_print = f"Hypervolume: {hypervolume} Crowding Dis: {average_crowding_dist}"
print(text_to_print)
# print(archive.solution_vectors)
with open(f"{folder_to_keep}/result.txt", "w") as file:
    file.write(text_to_print)
### visualization of solution
# print(positions)
# print(archive.solution_vectors)

padding = 5
font_size = 12
line_width = 2
marker_size = 4

solutions = archive.solution_vectors
print(len(solutions))
print(len(archive.objective_vectors))

# print(solutions)

sorted_solutions = [np.argsort(arr) for arr in solutions]
unique_solutions = [np.array(arr) for arr in sorted_solutions]

sorted_scores = np.argsort(Y[:,0])
print(sorted_scores)
print(len(Y[:,0]))
print(len(unique_solutions))
Y = Y[sorted_scores]

sorted_unique_solutions = []
for i in sorted_scores:
    sorted_unique_solutions.append(unique_solutions[sorted_scores[i]])
unique_solutions = sorted_unique_solutions

print("Total unique solutions is : ", len(unique_solutions))


num_of_cols = 5
num_of_rows = int(np.ceil(len(unique_solutions) / num_of_cols))

#Obtain the ranking

for sol_idx in tqdm(range(len(unique_solutions))):
    plt.figure(figsize=(12,12))


    sorted_positions = unique_solutions[sol_idx]  # Arranges in ascending order
    rankings = np.empty_like(sorted_positions)
    rankings[sorted_positions] = np.arange(len(position_scores), 0, -1)
    # print(rankings)

    order_to_travel = []
    order_to_travel_idx = []
    #Order the coordinates accordingly
    for i in range(len(rankings)):
        cur_pos = i + 1
        cur_coords_idx = np.argwhere(rankings == cur_pos)[0][0]
        order_to_travel_idx.append(cur_coords_idx)

        cur_coords = np.array(positions[cur_coords_idx])
        order_to_travel.append(cur_coords)

    x_coordinates, y_coordinates = zip(*order_to_travel)

    plt.plot(x_coordinates, y_coordinates, color='blue', marker='o', linewidth=line_width, zorder=1)
    plt.xlim(min(x_coordinates) - padding, max(x_coordinates) + padding)  # Adding space to the x-axis
    plt.ylim(min(y_coordinates) - padding, max(y_coordinates) + padding)  # Adding space to the y-axis

    cur_pt = 0
    for i in range(len(x_coordinates)):
        x = x_coordinates[i]
        y = y_coordinates[i]

        if i == 0:
            color_point = 'green'
        elif i == len(x_coordinates) - 1:
            color_point = 'green'
        else:
            color_point = 'orange'

        pt_idx = order_to_travel_idx[cur_pt]
        plt.scatter(x, y, label=f"({x}, {y}), pos {i+1}, demand {demand[pt_idx]}", color=color_point, marker='o', s = 50, zorder=2)
        plt.annotate(f'{i+1} ', (x, y), textcoords="offset points",
                     xytext=(5, 5), ha='right', fontsize=font_size)
        cur_pt += 1

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Point Order")

    # Add title and labels
    plt.title(f'Solution {sol_idx+1}, Distance Score {Y[sol_idx,0]:.6f}, Demand Score  {Y[sol_idx,1]:.6f}')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # Show grid
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder_to_keep}/solution_{sol_idx+1}.png", dpi=300, bbox_inches="tight")  # Save as high-quality image

