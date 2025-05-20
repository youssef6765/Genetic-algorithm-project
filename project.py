import numpy as np
import random

# Parameters
POP_SIZE = 50
MAX_GENERATIONS = 200
CROSSOVER_RATE = 0.7
ELITE_SIZE = 2

def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        chromosome = [
            round(random.uniform(0, 3), 2),    # a: Vegetable protein
            round(random.uniform(0, 3), 2),    # b: Animal protein
            round(random.uniform(0, 2), 2),    # c: Carbohydrates
            round(random.uniform(0, 2), 2),    # d: Vegetables
            round(random.uniform(0, 2), 2),    # e: Fruit
            round(random.uniform(0.5, 2), 2),  # f: Water (minimum 0.5)
            round(random.uniform(0, 1.5), 2),  # g: Healthy fats
            round(random.uniform(0, 3), 2),    # h: Dairy or alternatives
            round(random.uniform(0, 3), 2),    # i: Legumes
            round(random.uniform(0, 2), 2)     # j: Whole grains
        ]
        population.append(chromosome)
    return population

def objective_function(chromosome):
    a, b, c, d, e, f, g, h, i, j = chromosome
    sum_value = 3*a + 3*b + 8*c + 5*d + 5*e + 8*f + 4*g + 3*h + 4*i + 6*j
    return abs(sum_value - 24)

def calculate_penalties(chromosome):
    a, b, c, d, e, f, g, h, i, j = chromosome
    total = a + b + c + d + e + f + g + h + i + j
    penalties = 0

    # Protein balance: 20–30%
    protein_ratio = (a + b + h + i) / total if total > 0 else 0
    if protein_ratio < 0.20:
        penalties += 10 * (0.20 - protein_ratio)
    elif protein_ratio > 0.30:
        penalties += 10 * (protein_ratio - 0.30)

    # Carbohydrate limit: 50–60%
    carb_ratio = (c + j) / total if total > 0 else 0
    if carb_ratio < 0.50:
        penalties += 10 * (0.50 - carb_ratio)
    elif carb_ratio > 0.60:
        penalties += 10 * (carb_ratio - 0.60)

    # Fat limit: 10–15%
    fat_ratio = g / total if total > 0 else 0
    if fat_ratio < 0.10:
        penalties += 10 * (0.10 - fat_ratio)
    elif fat_ratio > 0.15:
        penalties += 10 * (fat_ratio - 0.15)

    # Minimum water: f >= 0.5
    if f < 0.5:
        penalties += 10 * (0.5 - f)

    # Diversity: At least 6 non-zero components
    non_zero_count = sum(1 for x in chromosome if x > 0)
    if non_zero_count < 6:
        penalties += 10 * (6 - non_zero_count)

    # Vitamin C: >= 30 mg
    vitamin_c = 20*d + 30*e + 5*h
    if vitamin_c < 30:
        penalties += 10 * (30 - vitamin_c) / 30

    # Sodium limit: <= 500 mg
    sodium = 100*b + 80*h + 50*j
    if sodium > 500:
        penalties += 10 * (sodium - 500) / 100

    # Vegetarian preference: Animal-based <= 20%
    animal_ratio = (b + h) / total if total > 0 else 0
    if animal_ratio > 0.20:
        penalties += 10 * (animal_ratio - 0.20)

    return penalties

def fitness_function(chromosome):
    return 1 / (objective_function(chromosome) + calculate_penalties(chromosome) + 1)

def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    selected = []
    for _ in range(POP_SIZE - ELITE_SIZE):
        r = random.random()
        cum_prob = 0
        for i, prob in enumerate(probabilities):
            cum_prob += prob
            if r <= cum_prob:
                selected.append(population[i].copy())
                break
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1.copy(), parent2.copy()

# Fuzzy Logic Functions
def membership_generation(generation_norm):
    # Early: [0, 0, 0.5]
    early = max(0, min((0.5 - generation_norm) / 0.5, 1)) if generation_norm <= 0.5 else 0
    # Middle: [0, 0.5, 1]
    middle = max(0, min((generation_norm - 0) / 0.5, (1 - generation_norm) / 0.5)) if 0 <= generation_norm <= 1 else 0
    # Late: [0.5, 1, 1]
    late = max(0, min((generation_norm - 0.5) / 0.5, 1)) if generation_norm >= 0.5 else 0
    return early, middle, late

def membership_mutation_rate(mr):
    # Low: [0, 0.02, 0.03]
    low = max(0, min((0.03 - mr) / (0.03 - 0.02), (mr - 0) / (0.02 - 0))) if 0 <= mr <= 0.03 else 0
    # Medium: [0.03, 0.05, 0.07]
    medium = max(0, min((mr - 0.03) / (0.05 - 0.03), (0.07 - mr) / (0.07 - 0.05))) if 0.03 <= mr <= 0.07 else 0
    # High: [0.07, 0.11, 0.15]
    high = max(0, min((mr - 0.07) / (0.11 - 0.07), (0.15 - mr) / (0.15 - 0.11))) if 0.07 <= mr <= 0.15 else 0
    return low, medium, high

def fuzzy_mutation_rate(generation):
    generation_norm = generation / MAX_GENERATIONS
    early, middle, late = membership_generation(generation_norm)
    
    # Fuzzy rules
    high_membership = early  # Rule 1: Early -> High
    medium_membership = middle  # Rule 2: Middle -> Medium
    low_membership = late  # Rule 3: Late -> Low
    
    # Defuzzification using centroid method
    mr_values = np.linspace(0, 0.15, 100)
    weighted_sum = 0
    total_weight = 0
    
    for mr in mr_values:
        low, medium, high = membership_mutation_rate(mr)
        # Combine memberships using max (Mamdani inference)
        membership = max(min(low, low_membership), min(medium, medium_membership), min(high, high_membership))
        weighted_sum += mr * membership
        total_weight += membership
    
    return weighted_sum / total_weight if total_weight > 0 else 0.05  # Fallback to 0.05 if no membership

def mutate(chromosome, mutation_rate):
    ranges = [3, 3, 2, 2, 2, 2, 1.5, 3, 3, 2]
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            min_val = 0.5 if i == 5 else 0
            chromosome[i] = round(random.uniform(min_val, ranges[i]), 2)
    return chromosome

def compute_sum(chromosome):
    a, b, c, d, e, f, g, h, i, j = chromosome
    return 3*a + 3*b + 8*c + 5*d + 5*e + 8*f + 4*g + 3*h + 4*i + 6*j

def genetic_algorithm():
    population = initialize_population()
    best_solutions = []
    
    for generation in range(MAX_GENERATIONS):
        # Compute fuzzy mutation rate for this generation
        mutation_rate = fuzzy_mutation_rate(generation)
        
        fitnesses = [fitness_function(chrom) for chrom in population]
        
        for i, chrom in enumerate(population):
            if objective_function(chrom) == 0 and calculate_penalties(chrom) == 0:
                best_solutions.append(chrom.copy())
                if len(best_solutions) >= 2:
                    return best_solutions[:2]
        
        population.sort(key=lambda x: objective_function(x) + calculate_penalties(x))
        new_population = population[:ELITE_SIZE]
        
        selected = roulette_wheel_selection(population, fitnesses)
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        
        population = [mutate(chrom.copy(), mutation_rate) for chrom in new_population[:POP_SIZE]]
    
    population.sort(key=lambda x: objective_function(x) + calculate_penalties(x))
    return population[:2]

def main():
    solutions = genetic_algorithm()
    print("\nOptimal Solutions Found:")
    for i, sol in enumerate(solutions, 1):
        print(f"Solution {i}:")
        components = [
            "Vegetable Protein (a)", "Animal Protein (b)", "Carbohydrate (c)",
            "Vegetable (d)", "Fruit (e)", "Water (f)", "Healthy Fats (g)",
            "Dairy or Alternatives (h)", "Legumes (i)", "Whole Grains (j)"
        ]
        for j, (comp, val) in enumerate(zip(components, sol)):
            print(f"{comp} = {val} servings")
        obj_val = objective_function(sol)
        penalties = calculate_penalties(sol)
        sum_value = compute_sum(sol)
        print(f"Sum Value: {sum_value:.2f}")
        print(f"Objective Function Value: {obj_val:.2f}")
        print(f"Penalties: {penalties:.2f}")
        if penalties > 0:
            print("Constraint Violations Detected:")
            a, b, c, d, e, f, g, h, i, j = sol
            total = a + b + c + d + e + f + g + h + i + j
            protein_ratio = (a + b + h + i) / total if total > 0 else 0
            carb_ratio = (c + j) / total if total > 0 else 0
            fat_ratio = g / total if total > 0 else 0
            non_zero_count = sum(1 for x in sol if x > 0)
            calories = 50*a + 70*b + 80*c + 25*d + 60*e + 90*g + 60*h + 50*i + 70*j
            vitamin_c = 20*d + 30*e + 5*h
            sodium = 100*b + 80*h + 50*j
            animal_ratio = (b + h) / total if total > 0 else 0
            if protein_ratio < 0.20 or protein_ratio > 0.30:
                print(f" - Protein Ratio: {protein_ratio:.2f} (should be 0.20–0.30)")
            if carb_ratio < 0.50 or carb_ratio > 0.60:
                print(f" - Carbohydrate Ratio: {carb_ratio:.2f} (should be 0.50–0.60)")
            if fat_ratio < 0.10 or fat_ratio > 0.15:
                print(f" - Fat Ratio: {fat_ratio:.2f} (should be 0.10–0.15)")
            if f < 0.5:
                print(f" - Water: {f:.2f} (should be >= 0.5)")
            if non_zero_count < 6:
                print(f" - Non-zero Components: {non_zero_count} (should be >= 6)")
            if calories < 400 or calories > 600:
                print(f" - Calories: {calories:.2f} kcal (should be 400–600)")
            if vitamin_c < 30:
                print(f" - Vitamin C: {vitamin_c:.2f} mg (should be >= 30)")
            if sodium > 500:
                print(f" - Sodium: {sodium:.2f} mg (should be <= 500)")
            if animal_ratio > 0.20:
                print(f" - Animal Ratio: {animal_ratio:.2f} (should be <= 0.20)")
        print()

if __name__ == "__main__":
    main()