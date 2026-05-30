# Optimization Models for Unrelated Parallel Machine Scheduling Problem

## Overview

This repository contains the implementation developed for my Bachelor's Thesis (TFG), focused on solving a scheduling problem in **Unrelated Parallel Machines with sequence-dependent setup times**.

The objective is to assign and sequence a set of tasks among multiple machines while minimizing the **makespan** (the completion time of the busiest machine).

Three different approaches are implemented and compared:

* **Mixed Integer Linear Programming (MILP)** – Exact optimization model.
* **Ant Colony Optimization (ACO)** – Swarm intelligence metaheuristic.
* **Genetic Algorithm (GA)** – Evolutionary optimization approach.

Additionally, a random instance generator is provided to create benchmark datasets for computational experiments.

---

## Problem Description

Given:

* A set of tasks (N)
* A set of machines (M)
* Machine-dependent processing times
* Sequence-dependent setup times
* Eligibility constraints (not every machine can process every task)

The goal is to:

1. Assign every task to exactly one machine.
2. Determine the processing sequence of tasks on each machine.
3. Minimize the makespan.

The cost of executing task (j) immediately after task (i) on machine (m) is:

[
C_{mij} = p_{mj} + s_{mij}
]

where:

* (p_{mj}) is the processing time of task (j) on machine (m)
* (s_{mij}) is the setup time required when task (j) follows task (i)

---

## Repository Structure

```text
├── MILP.py
├── ant_colony_optimization.py
├── genetic_algorithm.py
├── generator.py
└── README.md
```

### MILP.py

Implementation of an exact Mixed Integer Linear Programming formulation using:

* CVXPY
* CPLEX

Features:

* Binary assignment and sequencing variables.
* Subtour elimination constraints.
* Makespan minimization.
* Exact optimal solutions for small and medium instances.

---

### ant_colony_optimization.py

Implementation of an Ant Colony Optimization algorithm.

Main components:

* Route construction based on pheromone trails.
* Visibility heuristic using inverse costs.
* Pheromone evaporation and reinforcement.
* Makespan-based solution evaluation.

Suitable for obtaining high-quality solutions in larger instances where exact methods become computationally expensive.

---

### genetic_algorithm.py

Implementation of a Genetic Algorithm using DEAP.

Includes:

* Custom individual representation.
* Feasibility repair mechanisms.
* Tournament selection.
* Crossover and mutation operators.
* Makespan-based fitness evaluation.

Designed to efficiently explore large solution spaces.

---

### generator.py

Random instance generator.

Generates:

* Machine-task eligibility matrices.
* Processing times.
* Sequence-dependent setup times.
* Combined cost matrices.

This module allows the creation of reproducible benchmark instances for experimental evaluation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/itsasoarizti/Optimization-Models-for-Unrelated-Parallel-Machine-Problem.git
cd Optimization-Models-for-Unrelated-Parallel-Machine-Problem
```

Install dependencies:

```bash
pip install numpy cvxpy deap
```

For the MILP model, a compatible solver such as IBM CPLEX is required.

---

## Example Usage

Generate a random instance:

```python
from generator import generator

m = 5
n = 20

C = generator(
    m=m,
    n=n,
    processing_time_gap=(1, 50),
    configuration_time_gap=(1, 20)
)
```

Solve with MILP:

```python
from MILP import MILP

makespan, routes = MILP(n, m, C)
```

Solve with Ant Colony Optimization:

```python
from ant_colony_optimization import ACO

makespan = ACO(n, m, C)
```

Solve with Genetic Algorithm:

```python
from genetic_algorithm import GEN

makespan, solution = GEN(n, m, C)
```

---

## Research Objective

The purpose of this project is to analyze the trade-off between:

* Solution quality.
* Computational time.
* Scalability.

by comparing exact optimization techniques against metaheuristic approaches for complex scheduling problems.

---

## Technologies

* Python
* NumPy
* CVXPY
* CPLEX
* DEAP

---

## Author

Bachelor's Thesis (TFG)

Degree in Mathematics
Author: Itsaso Ariztimuño Cenoz
