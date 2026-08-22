"""
VRP com CAPACIDADE DE CARGA e PONTOS DE DIVISÃO EVOLUÍVEIS (versão corrigida).

Problema identificado na primeira versão (vrp_capacity_fixed_splits.py): dividir a
permutação sempre em pedaços de tamanho fixo (ex: 5, 5, 5 cidades) impede o
algoritmo de balancear a demanda entre veículos -- ele só pode reordenar
QUAIS cidades vão pra cada veículo, nunca QUANTAS. Se não existir nenhuma
divisão de tamanho fixo que respeite a capacidade, o algoritmo fica preso
numa solução inválida sem conseguir evoluir para uma válida.

Solução: os pontos de corte (splits) que dividem a permutação em N_VEHICLES
rotas agora fazem parte do cromossomo e evoluem junto com a ordem das
cidades. Isso permite rotas de tamanhos diferentes, dando ao algoritmo
liberdade real para balancear a carga.

Representação do indivíduo: tupla (permutation, splits)
- permutation: lista de cidades, igual antes
- splits: lista de N_VEHICLES-1 índices que dividem a permutation em rotas
  Ex: para 15 cidades e 3 veículos, splits=[4, 9] gera rotas de tamanho
  [4, 5, 6] (0:4, 4:9, 9:15)
"""

import random
from typing import List, Tuple, Dict

from genetic_algorithm import (
    calculate_distance,
    generate_random_population,
    order_crossover,
    mutate,
    default_problems,
)
from vrp_capacity_fixed_splits import generate_demands, calculate_route_demand
from vrp import calculate_route_distance


Individual = Tuple[List[Tuple[float, float]], List[int]]


def generate_random_splits(n_cities: int, n_vehicles: int, rng: random.Random) -> List[int]:
    """
    Gera N_VEHICLES-1 pontos de corte distintos e ordenados, dividindo a
    permutação em N_VEHICLES pedaços (nenhum pedaço fica vazio).
    """
    if n_vehicles == 1:
        return []
    # pontos possíveis: 1 até n_cities-1 (garante que nenhuma rota fica vazia)
    possible_points = list(range(1, n_cities))
    splits = sorted(rng.sample(possible_points, n_vehicles - 1))
    return splits


def split_into_routes_variable(chromosome: List[Tuple[float, float]], splits: List[int]) -> List[List[Tuple[float, float]]]:
    """Divide a permutação em rotas de tamanhos variáveis, usando os pontos de corte."""
    points = [0] + list(splits) + [len(chromosome)]
    return [chromosome[points[i]:points[i + 1]] for i in range(len(points) - 1)]


def generate_random_population_vrp(cities: List[Tuple[float, float]], population_size: int, n_vehicles: int, rng: random.Random) -> List[Individual]:
    """Gera a população inicial: cada indivíduo tem uma permutação aleatória + splits aleatórios."""
    population = []
    for _ in range(population_size):
        permutation = rng.sample(cities, len(cities))
        splits = generate_random_splits(len(cities), n_vehicles, rng)
        population.append((permutation, splits))
    return population


def crossover_splits(splits1: List[int], splits2: List[int], n_cities: int, rng: random.Random) -> List[int]:
    """
    Combina os splits de dois pais: para cada ponto de corte, sorteia de
    qual pai herdar. Depois remove duplicatas e garante que o número de
    splits está correto, completando com pontos aleatórios se necessário.
    """
    n_splits_needed = len(splits1)  # mesmo número em ambos os pais (n_vehicles-1)

    combined = set()
    for i in range(n_splits_needed):
        chosen = splits1[i] if rng.random() < 0.5 else splits2[i]
        combined.add(chosen)

    # se sobrou menos pontos que o necessário (por causa de duplicatas),
    # completa com pontos aleatórios válidos
    possible_points = set(range(1, n_cities))
    available = list(possible_points - combined)
    rng.shuffle(available)
    while len(combined) < n_splits_needed and available:
        combined.add(available.pop())

    return sorted(combined)


def mutate_splits(splits: List[int], n_cities: int, mutation_probability: float, rng: random.Random) -> List[int]:
    """
    Com uma certa probabilidade, desloca um ponto de corte aleatoriamente
    (pequeno deslocamento), mantendo os splits válidos (únicos, ordenados,
    dentro dos limites).
    """
    if not splits or rng.random() >= mutation_probability:
        return splits[:]

    new_splits = set(splits)
    idx = rng.randrange(len(splits))
    original_value = splits[idx]

    shift = rng.choice([-2, -1, 1, 2])
    new_value = original_value + shift
    new_value = max(1, min(n_cities - 1, new_value))

    new_splits.discard(original_value)
    if new_value not in new_splits:
        new_splits.add(new_value)
    else:
        new_splits.add(original_value)  # não conseguiu mover, mantém original

    return sorted(new_splits)


def calculate_fitness_vrp_capacity_v2(
    individual: Individual,
    demands: Dict[Tuple[float, float], int],
    vehicle_capacity: int,
    penalty_factor: float = 50.0,
) -> float:
    chromosome, splits = individual
    routes = split_into_routes_variable(chromosome, splits)

    total_distance = 0.0
    total_penalty = 0.0

    for route in routes:
        total_distance += calculate_route_distance(route)
        route_demand = calculate_route_demand(route, demands)
        excess = max(0, route_demand - vehicle_capacity)
        total_penalty += excess * penalty_factor

    return total_distance + total_penalty


def sort_population_vrp(population: List[Individual], fitness: List[float]) -> Tuple[List[Individual], List[float]]:
    combined = list(zip(population, fitness))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*combined_sorted)
    return list(sorted_population), list(sorted_fitness)


if __name__ == '__main__':
    N_CITIES = 15
    N_VEHICLES = 3
    POPULATION_SIZE = 100
    N_GENERATIONS = 150
    MUTATION_PROBABILITY = 0.3
    SPLIT_MUTATION_PROBABILITY = 0.3
    VEHICLE_CAPACITY = 20
    PENALTY_FACTOR = 50.0
    DEMAND_SEED = 42

    rng = random.Random(1)  # seed geral para reprodutibilidade da população/operadores

    cities_locations = default_problems[N_CITIES]
    demands = generate_demands(cities_locations, min_demand=1, max_demand=10, seed=DEMAND_SEED)

    print(f"Capacidade por veículo: {VEHICLE_CAPACITY}")
    print(f"Demanda total: {sum(demands.values())} | "
          f"Capacidade total da frota: {VEHICLE_CAPACITY * N_VEHICLES}\n")

    population = generate_random_population_vrp(cities_locations, POPULATION_SIZE, N_VEHICLES, rng)

    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [
            calculate_fitness_vrp_capacity_v2(ind, demands, VEHICLE_CAPACITY, PENALTY_FACTOR)
            for ind in population
        ]

        population, population_fitness = sort_population_vrp(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        if generation % 10 == 0 or generation == N_GENERATIONS - 1:
            chromosome, splits = best_solution
            routes = split_into_routes_variable(chromosome, splits)
            route_demands = [calculate_route_demand(r, demands) for r in routes]
            route_sizes = [len(r) for r in routes]
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f} "
                  f"| Tamanhos: {route_sizes} | Demandas: {route_demands}")

        new_population = [population[0]]  # ELITISM

        while len(new_population) < POPULATION_SIZE:
            (parent1_chrom, parent1_splits), (parent2_chrom, parent2_splits) = random.choices(population[:10], k=2)

            child_chrom = order_crossover(parent1_chrom, parent2_chrom)
            child_chrom = mutate(child_chrom, MUTATION_PROBABILITY)

            child_splits = crossover_splits(parent1_splits, parent2_splits, N_CITIES, rng)
            child_splits = mutate_splits(child_splits, N_CITIES, SPLIT_MUTATION_PROBABILITY, rng)

            new_population.append((child_chrom, child_splits))

        population = new_population

    print("\n--- Resultado final ---")
    final_chrom, final_splits = best_solutions[-1]
    final_routes = split_into_routes_variable(final_chrom, final_splits)
    print(f"Melhor fitness (distância + penalidade): {best_fitness_values[-1]:.2f}\n")

    total_distance_final = 0.0
    all_valid = True
    for i, route in enumerate(final_routes):
        dist = calculate_route_distance(route)
        demand = calculate_route_demand(route, demands)
        total_distance_final += dist
        valid = demand <= VEHICLE_CAPACITY
        all_valid = all_valid and valid
        status = "OK" if valid else f"EXCEDE em {demand - VEHICLE_CAPACITY}"
        print(f"Veículo {i+1}: {len(route)} paradas | distância = {dist:.2f} "
              f"| demanda = {demand}/{VEHICLE_CAPACITY} [{status}]")

    print(f"\nDistância total real (sem penalidade): {total_distance_final:.2f}")
    print(f"Todas as rotas respeitam a capacidade? {'SIM' if all_valid else 'NÃO'}")
