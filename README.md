# Genetic-algorithm-project

# Dietary Plan Optimization with Genetic Algorithm

This project implements a genetic algorithm with fuzzy logic to optimize a dietary plan based on nutritional constraints. The goal is to find a combination of food servings that meets a target nutritional sum (24 units) while satisfying constraints on macronutrient ratios, water intake, dietary diversity, vitamin C, sodium, and vegetarian preferences. The algorithm uses a population-based approach to evolve solutions over generations, incorporating fuzzy logic to dynamically adjust the mutation rate.

## Problem Description

The dietary plan consists of 10 components, each representing servings of a specific food group:
- **a**: Vegetable Protein (0–3 servings)
- **b**: Animal Protein (0–3 servings)
- **c**: Carbohydrates (0–2 servings)
- **d**: Vegetables (0–2 servings)
- **e**: Fruit (0–2 servings)
- **f**: Water (0.5–2 servings)
- **g**: Healthy Fats (0–1.5 servings)
- **h**: Dairy or Alternatives (0–3 servings)
- **i**: Legumes (0–3 servings)
- **j**: Whole Grains (0–2 servings)

### Objective
Minimize the absolute difference between the weighted sum of servings (`3a + 3b + 8c + 5d + 5e + 8f + 4g + 3h + 4i + 6j`) and the target value of 24.

### Constraints
- **Protein Ratio**: Total protein (a + b + h + i) should be 20–30% of total servings.
- **Carbohydrate Ratio**: Carbohydrates (c + j) should be 50–60% of total servings.
- **Fat Ratio**: Healthy fats (g) should be 10–15% of total servings.
- **Water**: At least 0.5 servings (f >= 0.5).
- **Diversity**: At least 6 non-zero components.
- **Vitamin C**: At least 30 mg (20d + 30e + 5h >= 30).
- **Sodium**: No more than 500 mg (100b + 80h + 50j <= 500).
- **Vegetarian Preference**: Animal-based components (b + h) should be <= 20% of total servings.

Penalties are applied for violating these constraints, and the fitness function is defined as `1 / (objective_function + penalties + 1)`.

## Algorithm Details

### Genetic Algorithm
- **Population Size**: 50 chromosomes.
- **Generations**: Up to 200 generations or until two optimal solutions (objective = 0, penalties = 0) are found.
- **Elite Size**: 2 (best chromosomes preserved each generation).
- **Selection**: Roulette wheel selection based on fitness.
- **Crossover**: Single-point crossover with a rate of 0.7.
- **Mutation**: Random mutation within component ranges, with a dynamic mutation rate determined by fuzzy logic.

### Fuzzy Logic for Mutation Rate
The mutation rate is adjusted based on the current generation using fuzzy logic:
- **Input**: Normalized generation (generation / MAX_GENERATIONS).
- **Membership Functions**:
  - Generation: Early (0–0.5), Middle (0–1), Late (0.5–1).
  - Mutation Rate: Low (0–0.03), Medium (0.03–0.07), High (0.07–0.15).
- **Rules**:
  - If generation is Early, mutation rate is High.
  - If generation is Middle, mutation rate is Medium.
  - If generation is Late, mutation rate is Low.
- **Defuzzification**: Centroid method to compute the mutation rate.

### Functions
- `initialize_population()`: Generates a population of random chromosomes within specified ranges.
- `objective_function()`: Computes the absolute difference from the target sum (24).
- `calculate_penalties()`: Applies penalties for constraint violations.
- `fitness_function()`: Calculates fitness as the inverse of objective value plus penalties.
- `roulette_wheel_selection()`: Selects chromosomes based on fitness probabilities.
- `crossover()`: Performs single-point crossover between two parents.
- `mutate()`: Applies random mutations to chromosomes based on the fuzzy mutation rate.
- `membership_generation()` and `membership_mutation_rate()`: Define fuzzy membership functions.
- `fuzzy_mutation_rate()`: Computes the mutation rate using fuzzy logic.
- `compute_sum()`: Calculates the weighted sum of servings.
- `genetic_algorithm()`: Runs the main genetic algorithm loop.
- `main()`: Executes the algorithm and prints the optimal solutions with their details.

## Requirements

To run the code, install the required Python packages:
```bash
pip install numpy
