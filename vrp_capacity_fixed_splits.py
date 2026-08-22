"""
[VERSÃO COM LIMITAÇÃO CONHECIDA -- ver vrp_capacity.py para a versão corrigida]

Extensão do VRP com restrição de CAPACIDADE DE CARGA por veículo, usando
divisão de rotas em PEDAÇOS DE TAMANHO FIXO (splits fixos).

LIMITAÇÃO: como o tamanho de cada rota é sempre fixo (ex: sempre 5 cidades
por veículo em um cenário de 15 cidades / 3 veículos), o algoritmo só pode
reordenar QUAIS cidades vão pra cada veículo, nunca QUANTAS. Se não existir
nenhuma divisão de tamanho fixo que respeite a capacidade dado o conjunto de
demandas, o algoritmo fica preso numa solução inválida sem conseguir evoluir
para uma válida. Esse arquivo é mantido no repositório como registro da
primeira tentativa e do raciocínio que levou à correção em vrp_capacity.py
(que usa pontos de corte evoluíveis, permitindo rotas de tamanhos variáveis).

Baseado em vrp.py (múltiplos veículos, cromossomo como permutação única
dividida em N_VEHICLES rotas contíguas). Aqui adicionamos:

- Uma demanda (peso/volume simulado) para cada cidade, representando o
  pedido de medicamentos/insumos daquele ponto de entrega.
- Uma capacidade máxima de carga por veículo.
- Penalização no fitness quando uma rota excede a capacidade do veículo,
  em vez de proibir a rota completamente. Isso evita que o crossover e a
  mutação precisem de lógica especial para "reparar" indivíduos inválidos:
  o próprio processo de seleção natural tende a eliminar, ao longo das
  gerações, as rotas que violam a capacidade, pois elas ficam com fitness
  pior (maior = pior, já que fitness aqui é uma distância/custo).
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
from vrp import split_into_routes, calculate_route_distance


def generate_demands(cities: List[Tuple[float, float]], min_demand: int = 1, max_demand: int = 10, seed: int = None) -> Dict[Tuple[float, float], int]:
    """
    Gera uma demanda simulada para cada cidade (representa o volume/peso do
    pedido de medicamentos ou insumos daquele ponto de entrega).

    Em um cenário real, essa demanda viria de um sistema de pedidos --
    aqui é gerada aleatoriamente para fins de simulação e teste do algoritmo.
    """
    rng = random.Random(seed)
    return {city: rng.randint(min_demand, max_demand) for city in cities}


def calculate_route_demand(route: List[Tuple[float, float]], demands: Dict[Tuple[float, float], int]) -> int:
    """Soma a demanda de todas as cidades em uma rota."""
    return sum(demands[city] for city in route)


def calculate_fitness_vrp_capacity(
    chromosome: List[Tuple[float, float]],
    n_vehicles: int,
    demands: Dict[Tuple[float, float], int],
    vehicle_capacity: int,
    penalty_factor: float = 50.0,
) -> float:
    """
    Fitness = soma das distâncias de todas as rotas + penalidade por excesso
    de capacidade em qualquer rota.

    penalty_factor controla o quão "caro" é violar a capacidade. Um valor
    alto força o algoritmo a priorizar fortemente soluções válidas; um valor
    baixo permite mais exploração de soluções que violam a restrição, mas
    corre risco de a solução final ainda violar a capacidade.
    """
    routes = split_into_routes(chromosome, n_vehicles)

    total_distance = 0.0
    total_penalty = 0.0

    for route in routes:
        total_distance += calculate_route_distance(route)

        route_demand = calculate_route_demand(route, demands)
        excess = max(0, route_demand - vehicle_capacity)
        total_penalty += excess * penalty_factor

    return total_distance + total_penalty


def sort_population_vrp(population, fitness):
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
    VEHICLE_CAPACITY = 20          # capacidade máxima de carga por veículo
    PENALTY_FACTOR = 50.0          # "custo" de cada unidade de demanda excedente
    DEMAND_SEED = 42               # fixa a seed para reprodutibilidade dos testes

    cities_locations = default_problems[N_CITIES]
    demands = generate_demands(cities_locations, min_demand=1, max_demand=10, seed=DEMAND_SEED)

    print("Demandas geradas por cidade:")
    for city, demand in demands.items():
        print(f"  {city}: {demand}")
    print(f"\nCapacidade por veículo: {VEHICLE_CAPACITY}")
    print(f"Demanda total: {sum(demands.values())} | "
          f"Capacidade total da frota: {VEHICLE_CAPACITY * N_VEHICLES}\n")

    population = generate_random_population(cities_locations, POPULATION_SIZE)

    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [
            calculate_fitness_vrp_capacity(individual, N_VEHICLES, demands, VEHICLE_CAPACITY, PENALTY_FACTOR)
            for individual in population
        ]

        population, population_fitness = sort_population_vrp(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        if generation % 10 == 0 or generation == N_GENERATIONS - 1:
            routes = split_into_routes(best_solution, N_VEHICLES)
            route_demands = [calculate_route_demand(r, demands) for r in routes]
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f} "
                  f"| Demandas por rota: {route_demands}")

        new_population = [population[0]]  # ELITISM

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[:10], k=2)
            child1 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            new_population.append(child1)

        population = new_population

    print("\n--- Resultado final ---")
    final_routes = split_into_routes(best_solutions[-1], N_VEHICLES)
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
