# Gravitational search algorithm

import math
import random

MAXIMIZATION = 0
MINIMIZATION = 1


class Agent:
    def __init__(self, mass, positions, velocities, total_gravitational_fields, fitness):
        self.mass = mass
        self.positions = positions
        self.velocities = velocities
        self.total_gravitational_fields = total_gravitational_fields
        self.fitness = fitness


class Iteration:
    def __init__(self, agents):
        self.agents = agents


class GSA:
    def __init__(self, fitness_func, optimization_type, number_of_agents, number_of_iterations, number_of_dimensions,
                 min_search_value, max_search_value):
        self.fitness_func = fitness_func
        self.optimization_type = optimization_type
        self.number_of_agents = number_of_agents
        self.max_iterations = number_of_iterations
        self.number_of_dimensions = number_of_dimensions
        self.min_search_value = min_search_value
        self.max_search_value = max_search_value
        self.iterations = []

    def distance(self, agent1, agent2):
        distance = 0
        for i in range(len(agent1.positions)):
            distance += (agent1.positions[i] - agent2.positions[i]) ** 2
        return math.sqrt(distance)

    @property
    def gravitational_constant(self):
        return 100.0 * math.exp(-20.0 * len(self.iterations) / self.max_iterations)

    def fitness(self, iteration):
        for agent in iteration.agents:
            agent.fitness = self.fitness_func(agent.positions)

    def random_agents(self):
        agents = []
        for i in range(self.number_of_agents):
            positions = []
            velocities = [0 for _ in range(self.number_of_dimensions)]
            total_gravitational_fields = [0 for _ in range(self.number_of_dimensions)]
            for j in range(self.number_of_dimensions):
                positions.append(random.uniform(self.min_search_value, self.max_search_value))
            agents.append(Agent(1, positions, velocities, total_gravitational_fields, 0))
        return agents

    def mass(self, iteration, best_fitness, worst_fitness):
        for agent in iteration.agents:
            agent.mass = (agent.fitness - worst_fitness) / (best_fitness - worst_fitness)

        total_mass = sum([agent.mass for agent in iteration.agents])
        for agent in iteration.agents:
            agent.mass /= total_mass

    def gravitational_field(self, iteration, gravitational_constant):
        epsilon = 0.001

        for i in range(
                int(self.number_of_agents - self.number_of_agents * float(len(self.iterations)) / self.max_iterations)):
            agent1 = iteration.agents[i]
            for j in range(len(iteration.agents)):
                agent2 = iteration.agents[j]
                if i == j:
                    continue
                distance = self.distance(agent1, agent2)
                for k in range(self.number_of_dimensions):
                    agent1.total_gravitational_fields[k] += random.random() * agent2.mass * (
                            agent2.positions[k] - agent1.positions[k]) / (distance + epsilon)

        for agent in iteration.agents:
            for k in range(self.number_of_dimensions):
                agent.total_gravitational_fields[k] *= gravitational_constant

    def move(self, iteration):
        for agent in iteration.agents:
            for i in range(self.number_of_dimensions):
                agent.velocities[i] = agent.total_gravitational_fields[i] + agent.velocities[i] * random.random()
                agent.positions[i] += agent.velocities[i]

    def solve(self):
        first_iteration = Iteration(self.random_agents())
        self.iterations = [first_iteration]
        current_iteration = first_iteration

        while len(self.iterations) <= self.max_iterations:
            for agent in current_iteration.agents:
                for i in range(len(agent.positions)):
                    if agent.positions[i] < self.min_search_value or agent.positions[i] > self.max_search_value:
                        agent.positions[i] = random.uniform(self.min_search_value, self.max_search_value)
            self.fitness(current_iteration)
            ordered_agents = []

            if self.optimization_type == MAXIMIZATION:
                ordered_agents = sorted(current_iteration.agents, key=lambda x: x.fitness, reverse=True)
            elif self.optimization_type == MINIMIZATION:
                ordered_agents = sorted(current_iteration.agents, key=lambda x: x.fitness)

            current_iteration.agents = ordered_agents
            best_fitness_agent = ordered_agents[0]
            worst_fitness_agent = ordered_agents[-1]
            self.mass(current_iteration, best_fitness_agent.fitness, worst_fitness_agent.fitness)

            gravitational_constant = self.gravitational_constant

            self.gravitational_field(current_iteration, gravitational_constant)

            # deep clone current iteration
            current_iteration_copy = Iteration([Agent(agent.mass, agent.positions.copy(), agent.velocities.copy(),
                                                      agent.total_gravitational_fields.copy(), agent.fitness) for agent
                                                in
                                                current_iteration.agents])
            self.iterations.append(current_iteration_copy)
            current_iteration = current_iteration_copy
            self.move(current_iteration)

            if len(self.iterations) % 100 == 0:
                print("Iteration: ", len(self.iterations), " Best fitness: ", best_fitness_agent.fitness,
                      " Best position: ", best_fitness_agent.positions)

        # return best agent
        return self.iterations[-1].agents[0]


def main():
    # a simple fitness function
    def fitness_func(positions):
        return (positions[0] - 1) ** 2

    gsa = GSA(fitness_func, optimization_type=MINIMIZATION, number_of_agents=100, number_of_iterations=500,
              number_of_dimensions=1, min_search_value=-10, max_search_value=10)
    best_agent = gsa.solve()
    print(best_agent.positions)


if __name__ == '__main__':
    main()
