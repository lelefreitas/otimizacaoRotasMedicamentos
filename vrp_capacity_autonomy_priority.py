"""
VRP com CAPACIDADE + AUTONOMIA + PRIORIDADE DE ENTREGAS.

Baseado em vrp_capacity_autonomy.py. Adiciona a última restrição obrigatória
do enunciado: prioridades diferentes para entregas (medicamentos críticos
vs. insumos regulares).

Abordagem: cada cidade recebe uma prioridade ("critico" ou "regular"). A
função fitness penaliza entregas críticas que ficam posicionadas tarde
dentro da rota do veículo -- ou seja, o algoritmo é incentivado a sequenciar
entregas críticas o quanto antes, minimizando o tempo até que medicamentos
urgentes cheguem ao destino.

penalidade_prioridade = posição_da_cidade_na_rota × priority_penalty_factor
                         (somente para cidades críticas; posição é 0-indexed,
                         então a primeira parada da rota tem penalidade 0)

Assim como capacidade e autonomia, essa penalidade é somada ao fitness base
(distância total) -- o algoritmo precisa equilibrar as três restrições ao
mesmo tempo.
"""

import random
from typing import List, Tuple, Dict

from genetic_algorithm import (
    order_crossover,
    mutate,
    default_problems,
)
from vrp_capacity_fixed_splits import generate_demands, calculate_route_demand
from vrp import calculate_route_distance
from vrp_capacity import (
    Individual,
    split_into_routes_variable,
    generate_random_population_vrp,
    crossover_splits,
    mutate_splits,
)

CRITICO = "critico"
REGULAR = "regular"


def generate_priorities(cities: List[Tuple[float, float]], critico_ratio: float = 0.3, seed: int = None) -> Dict[Tuple[float, float], str]:
    """
    Sorteia a prioridade de cada cidade: 'critico' (medicamento urgente) ou
    'regular' (insumo regular). critico_ratio controla a proporção esperada
    de entregas críticas (ex: 0.3 = ~30% das cidades são críticas).
    """
    rng = random.Random(seed)
    return {city: (CRITICO if rng.random() < critico_ratio else REGULAR) for city in cities}


def calculate_priority_penalty(route: List[Tuple[float, float]], priorities: Dict[Tuple[float, float], str], priority_penalty_factor: float) -> float:
    """
    Soma, para cada cidade CRÍTICA na rota, um custo proporcional à sua
    posição na sequência (quanto mais tarde, maior o custo). Cidades
    regulares não geram penalidade -- só entregas críticas atrasadas.
    """
    penalty = 0.0
    for position, city in enumerate(route):
        if priorities[city] == CRITICO:
            penalty += position * priority_penalty_factor
    return penalty


def calculate_fitness_vrp_full(
    individual: Individual,
    demands: Dict[Tuple[float, float], int],
    priorities: Dict[Tuple[float, float], str],
    vehicle_capacity: int,
    vehicle_autonomy: float,
    capacity_penalty_factor: float = 50.0,
    autonomy_penalty_factor: float = 5.0,
    priority_penalty_factor: float = 10.0,
) -> float:
    chromosome, splits = individual
    routes = split_into_routes_variable(chromosome, splits)

    total_distance = 0.0
    total_capacity_penalty = 0.0
    total_autonomy_penalty = 0.0
    total_priority_penalty = 0.0

    for route in routes:
        route_distance = calculate_route_distance(route)
        total_distance += route_distance

        route_demand = calculate_route_demand(route, demands)
        capacity_excess = max(0, route_demand - vehicle_capacity)
        total_capacity_penalty += capacity_excess * capacity_penalty_factor

        autonomy_excess = max(0, route_distance - vehicle_autonomy)
        total_autonomy_penalty += autonomy_excess * autonomy_penalty_factor

        total_priority_penalty += calculate_priority_penalty(route, priorities, priority_penalty_factor)

    return total_distance + total_capacity_penalty + total_autonomy_penalty + total_priority_penalty


def sort_population_vrp(population: List[Individual], fitness: List[float]) -> Tuple[List[Individual], List[float]]:
    combined = list(zip(population, fitness))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*combined_sorted)
    return list(sorted_population), list(sorted_fitness)


def average_critical_position(routes: List[List[Tuple[float, float]]], priorities: Dict[Tuple[float, float], str]) -> float:
    """Métrica auxiliar: posição média das entregas críticas dentro de suas rotas (menor é melhor)."""
    positions = []
    for route in routes:
        for position, city in enumerate(route):
            if priorities[city] == CRITICO:
                positions.append(position)
    return sum(positions) / len(positions) if positions else 0.0


if __name__ == '__main__':
    N_CITIES = 15
    N_VEHICLES = 3
    POPULATION_SIZE = 100
    N_GENERATIONS = 200
    MUTATION_PROBABILITY = 0.3
    SPLIT_MUTATION_PROBABILITY = 0.3
    VEHICLE_CAPACITY = 20
    VEHICLE_AUTONOMY = 450.0        # autonomia mais folgada aqui para isolar o efeito da prioridade
    CAPACITY_PENALTY_FACTOR = 100.0
    AUTONOMY_PENALTY_FACTOR = 5.0
    PRIORITY_PENALTY_FACTOR = 15.0
    DEMAND_SEED = 42
    PRIORITY_SEED = 7
    CRITICO_RATIO = 0.3

    rng = random.Random(1)

    cities_locations = default_problems[N_CITIES]
    demands = generate_demands(cities_locations, min_demand=1, max_demand=10, seed=DEMAND_SEED)
    priorities = generate_priorities(cities_locations, critico_ratio=CRITICO_RATIO, seed=PRIORITY_SEED)

    n_criticos = sum(1 for p in priorities.values() if p == CRITICO)
    print(f"Capacidade por veículo: {VEHICLE_CAPACITY} | Autonomia: {VEHICLE_AUTONOMY}")
    print(f"Entregas críticas: {n_criticos}/{N_CITIES}\n")

    population = generate_random_population_vrp(cities_locations, POPULATION_SIZE, N_VEHICLES, rng)

    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [
            calculate_fitness_vrp_full(
                ind, demands, priorities, VEHICLE_CAPACITY, VEHICLE_AUTONOMY,
                CAPACITY_PENALTY_FACTOR, AUTONOMY_PENALTY_FACTOR, PRIORITY_PENALTY_FACTOR,
            )
            for ind in population
        ]

        population, population_fitness = sort_population_vrp(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        if generation % 20 == 0 or generation == N_GENERATIONS - 1:
            chromosome, splits = best_solution
            routes = split_into_routes_variable(chromosome, splits)
            avg_pos = average_critical_position(routes, priorities)
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f} "
                  f"| Posição média das entregas críticas: {avg_pos:.2f}")

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
    print(f"Melhor fitness (distância + penalidades): {best_fitness_values[-1]:.2f}\n")

    total_distance_final = 0.0
    for i, route in enumerate(final_routes):
        dist = calculate_route_distance(route)
        demand = calculate_route_demand(route, demands)
        total_distance_final += dist

        criticas_na_rota = [(pos, city) for pos, city in enumerate(route) if priorities[city] == CRITICO]
        posicoes_criticas = [pos for pos, _ in criticas_na_rota]

        print(f"Veículo {i+1}: {len(route)} paradas | distância = {dist:.2f} | demanda = {demand} "
              f"| entregas críticas nas posições: {posicoes_criticas if posicoes_criticas else 'nenhuma'}")

    avg_pos_final = average_critical_position(final_routes, priorities)
    print(f"\nDistância total real (sem penalidades): {total_distance_final:.2f}")
    print(f"Posição média final das entregas críticas: {avg_pos_final:.2f} "
          f"(quanto mais próximo de 0, mais cedo elas são entregues)")
