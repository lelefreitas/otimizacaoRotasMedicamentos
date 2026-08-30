"""
Pipeline completo: roda o algoritmo genético (VRP com capacidade, autonomia
e prioridade), pega a melhor solução encontrada, e alimenta automaticamente
as funções de LLM (vrp_llm.py) com os dados REAIS dessa solução -- sem
dados de exemplo fixos no código.

Fluxo:
    1. Roda o GA (vrp_capacity_autonomy_priority.py) por N gerações
    2. Extrai a melhor solução final: rotas, demandas, prioridades, distâncias
    3. Para cada rota: gera instruções para o motorista via LLM
    4. Para a frota inteira: gera um relatório de eficiência via LLM
    5. Demonstra uma pergunta em linguagem natural sobre as rotas geradas

Este arquivo é o ponto de entrada "de verdade" do projeto -- reúne GA e LLM
num único fluxo executável, em vez de rodar os dois separadamente.
"""

import random

from genetic_algorithm import order_crossover, mutate, default_problems
from vrp_capacity_fixed_splits import generate_demands
from vrp import calculate_route_distance
from vrp_capacity import (
    split_into_routes_variable,
    generate_random_population_vrp,
    crossover_splits,
    mutate_splits,
)
from vrp_capacity_autonomy_priority import (
    generate_priorities,
    calculate_fitness_vrp_full,
    sort_population_vrp,
    average_critical_position,
)
from vrp_llm import (
    get_client,
    generate_driver_instructions,
    generate_efficiency_report,
    answer_question_about_routes,
)


def run_genetic_algorithm(
    n_cities: int,
    n_vehicles: int,
    population_size: int,
    n_generations: int,
    mutation_probability: float,
    split_mutation_probability: float,
    vehicle_capacity: int,
    vehicle_autonomy: float,
    capacity_penalty_factor: float,
    autonomy_penalty_factor: float,
    priority_penalty_factor: float,
    demand_seed: int,
    priority_seed: int,
    critico_ratio: float,
    rng_seed: int,
):
    """
    Roda o GA completo e retorna a melhor solução final, junto com os
    dados auxiliares (demandas, prioridades) necessários para o LLM
    interpretar essa solução depois.
    """
    rng = random.Random(rng_seed)

    cities_locations = default_problems[n_cities]
    demands = generate_demands(cities_locations, min_demand=1, max_demand=10, seed=demand_seed)
    priorities = generate_priorities(cities_locations, critico_ratio=critico_ratio, seed=priority_seed)

    population = generate_random_population_vrp(cities_locations, population_size, n_vehicles, rng)

    best_solution = None
    best_fitness = None

    for generation in range(n_generations):
        population_fitness = [
            calculate_fitness_vrp_full(
                ind, demands, priorities, vehicle_capacity, vehicle_autonomy,
                capacity_penalty_factor, autonomy_penalty_factor, priority_penalty_factor,
            )
            for ind in population
        ]

        population, population_fitness = sort_population_vrp(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        if generation % 20 == 0 or generation == n_generations - 1:
            chromosome, splits = best_solution
            routes = split_into_routes_variable(chromosome, splits)
            avg_pos = average_critical_position(routes, priorities)
            print(f"Geração {generation}: fitness = {best_fitness:.2f} | posição média críticas = {avg_pos:.2f}")

        new_population = [population[0]]  # ELITISM

        while len(new_population) < population_size:
            (parent1_chrom, parent1_splits), (parent2_chrom, parent2_splits) = random.choices(population[:10], k=2)

            child_chrom = order_crossover(parent1_chrom, parent2_chrom)
            child_chrom = mutate(child_chrom, mutation_probability)

            child_splits = crossover_splits(parent1_splits, parent2_splits, n_cities, rng)
            child_splits = mutate_splits(child_splits, n_cities, split_mutation_probability, rng)

            new_population.append((child_chrom, child_splits))

        population = new_population

    return best_solution, best_fitness, demands, priorities


if __name__ == '__main__':
    # -- Parâmetros do GA (mesmos valores já validados em execuções anteriores) --
    N_CITIES = 15
    N_VEHICLES = 3
    POPULATION_SIZE = 100
    N_GENERATIONS = 200
    MUTATION_PROBABILITY = 0.3
    SPLIT_MUTATION_PROBABILITY = 0.3
    VEHICLE_CAPACITY = 20
    VEHICLE_AUTONOMY = 450.0
    CAPACITY_PENALTY_FACTOR = 100.0
    AUTONOMY_PENALTY_FACTOR = 5.0
    PRIORITY_PENALTY_FACTOR = 15.0
    DEMAND_SEED = 42
    PRIORITY_SEED = 7
    CRITICO_RATIO = 0.3
    RNG_SEED = 1

    print("=" * 70)
    print("ETAPA 1: Rodando o algoritmo genético")
    print("=" * 70)

    best_solution, best_fitness, demands, priorities = run_genetic_algorithm(
        n_cities=N_CITIES, n_vehicles=N_VEHICLES, population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS, mutation_probability=MUTATION_PROBABILITY,
        split_mutation_probability=SPLIT_MUTATION_PROBABILITY, vehicle_capacity=VEHICLE_CAPACITY,
        vehicle_autonomy=VEHICLE_AUTONOMY, capacity_penalty_factor=CAPACITY_PENALTY_FACTOR,
        autonomy_penalty_factor=AUTONOMY_PENALTY_FACTOR, priority_penalty_factor=PRIORITY_PENALTY_FACTOR,
        demand_seed=DEMAND_SEED, priority_seed=PRIORITY_SEED, critico_ratio=CRITICO_RATIO, rng_seed=RNG_SEED,
    )

    final_chrom, final_splits = best_solution
    final_routes = split_into_routes_variable(final_chrom, final_splits)
    route_distances = [calculate_route_distance(route) for route in final_routes]

    print(f"\nMelhor fitness final: {best_fitness:.2f}")
    print(f"Número de rotas (veículos): {len(final_routes)}")
    for i, (route, dist) in enumerate(zip(final_routes, route_distances)):
        demand = sum(demands[c] for c in route)
        criticas = sum(1 for c in route if priorities[c] == "critico")
        print(f"  Veículo {i+1}: {len(route)} paradas | distância = {dist:.2f} | demanda = {demand} | críticas = {criticas}")

    print("\n" + "=" * 70)
    print("ETAPA 2: Gerando instruções para motoristas via LLM (Gemini)")
    print("=" * 70)

    client = get_client()
    if client is None:
        print("\n[AVISO] GEMINI_API_KEY não configurada -- rodando em modo dry-run (mostra os prompts, não chama a API real).\n")

    for i, route in enumerate(final_routes):
        print(f"\n--- Instruções para o motorista do Veículo {i+1} ---")
        print(generate_driver_instructions(client, route, priorities, demands, i + 1))

    print("\n" + "=" * 70)
    print("ETAPA 3: Gerando relatório de eficiência da frota via LLM")
    print("=" * 70)
    print(generate_efficiency_report(
        client, final_routes, priorities, demands, route_distances,
        vehicle_capacity=VEHICLE_CAPACITY, vehicle_autonomy=VEHICLE_AUTONOMY,
    ))

    print("\n" + "=" * 70)
    print("ETAPA 4: Demonstração de pergunta em linguagem natural")
    print("=" * 70)
    pergunta = "Qual veículo tem a rota mais longa e quantas entregas críticas ele carrega?"
    print(f"\nPergunta: {pergunta}\n")
    print(answer_question_about_routes(client, pergunta, final_routes, priorities, demands, route_distances))
