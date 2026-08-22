"""
Extensão do problema TSP para VRP (Vehicle Routing Problem) com múltiplos veículos.

Abordagem: mantém a representação de cromossomo como uma ÚNICA permutação de
cidades (igual ao TSP original) e reaproveita order_crossover e mutate sem
alterações. A diferença é apenas na hora de CALCULAR o fitness: a permutação
é dividida em N_VEHICLES pedaços contíguos, cada pedaço vira a rota de um
veículo, e o fitness final é a SOMA das distâncias de todas as rotas.

Isso reaproveita os operadores genéticos já validados no TSP (e com o bug do
order_crossover já corrigido), minimizando risco de introduzir bugs novos
nesta expansão.

Restrições NÃO incluídas nesta primeira versão (ficam para os próximos passos):
- capacidade de carga por veículo
- autonomia (distância máxima) por veículo
- prioridade de entregas
"""

import random
from typing import List, Tuple

from genetic_algorithm import (
    calculate_distance,
    generate_random_population,
    order_crossover,
    mutate,
    sort_population,
    default_problems,
)


def split_into_routes(chromosome: List[Tuple[float, float]], n_vehicles: int) -> List[List[Tuple[float, float]]]:
    """
    Divide uma permutação única de cidades em N_VEHICLES rotas contíguas,
    o mais equilibradas possível em tamanho.

    Exemplo: 10 cidades, 3 veículos -> rotas de tamanho [4, 3, 3]
    """
    n_cities = len(chromosome)
    base_size = n_cities // n_vehicles
    remainder = n_cities % n_vehicles

    routes = []
    start = 0
    for vehicle_idx in range(n_vehicles):
        # os primeiros 'remainder' veículos recebem uma cidade a mais,
        # para distribuir o resto de forma equilibrada
        size = base_size + (1 if vehicle_idx < remainder else 0)
        routes.append(chromosome[start:start + size])
        start += size

    return routes


def calculate_route_distance(route: List[Tuple[float, float]]) -> float:
    """
    Calcula a distância de UMA rota (sem retorno ao ponto de partida,
    diferente do TSP original que é cíclico). Faz mais sentido para
    entregas: o veículo não precisa necessariamente voltar ao início
    entre uma parada e outra dentro da mesma rota.

    Se a rota tiver 0 ou 1 cidade, a distância é 0 (nada a percorrer).
    """
    if len(route) < 2:
        return 0.0

    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])

    return distance


def calculate_fitness_vrp(chromosome: List[Tuple[float, float]], n_vehicles: int) -> float:
    """
    Fitness do VRP = soma das distâncias de todas as rotas (uma por veículo).

    Quanto menor, melhor (mesma lógica do TSP original).
    """
    routes = split_into_routes(chromosome, n_vehicles)
    return sum(calculate_route_distance(route) for route in routes)


def sort_population_vrp(population, fitness):
    """Idêntico ao sort_population do TSP -- reaproveitado para clareza semântica."""
    return sort_population(population, fitness)


if __name__ == '__main__':
    N_CITIES = 15
    N_VEHICLES = 3
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3

    cities_locations = default_problems[N_CITIES]

    # CREATE INITIAL POPULATION (reaproveita a função do TSP: cada indivíduo
    # ainda é só uma permutação de cidades, igual antes)
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [calculate_fitness_vrp(individual, N_VEHICLES) for individual in population]

        population, population_fitness = sort_population_vrp(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Generation {generation}: Best fitness = {best_fitness:.2f} "
              f"| Rotas: {[len(r) for r in split_into_routes(best_solution, N_VEHICLES)]}")

        new_population = [population[0]]  # ELITISM

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[:10], k=2)
            child1 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            new_population.append(child1)

        population = new_population

    print("\n--- Resultado final ---")
    print(f"Melhor fitness: {best_fitness_values[-1]:.2f}")
    final_routes = split_into_routes(best_solutions[-1], N_VEHICLES)
    for i, route in enumerate(final_routes):
        print(f"Veículo {i+1}: {len(route)} paradas, distância = {calculate_route_distance(route):.2f}")
