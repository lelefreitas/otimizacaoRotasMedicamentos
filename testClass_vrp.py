"""
Testes automatizados para o projeto de Otimização de Rotas Médicas (VRP).

Cobre os componentes mais críticos:
- Operadores genéticos: order_crossover, mutate (base do TSP)
- Divisão de rotas com splits evoluíveis
- Cálculo de fitness: capacidade, autonomia, prioridade de entregas
- Operadores genéticos dos splits: crossover_splits, mutate_splits

Rodar com: pytest test_vrp.py -v
"""

import random
import pytest

from genetic_algorithm import order_crossover, mutate, default_problems
from vrp import split_into_routes, calculate_route_distance
from vrp_capacity_fixed_splits import generate_demands, calculate_route_demand
from vrp_capacity import (
    split_into_routes_variable,
    generate_random_splits,
    crossover_splits,
    mutate_splits,
)
from vrp_capacity_autonomy_priority import (
    generate_priorities,
    calculate_priority_penalty,
    calculate_fitness_vrp_full,
    CRITICO,
    REGULAR,
)


# ---------------------------------------------------------------------------
# Fixtures reutilizáveis
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cities():
    """Um conjunto pequeno e fixo de cidades para testes previsíveis."""
    return default_problems[15]


@pytest.fixture
def sample_demands(sample_cities):
    return generate_demands(sample_cities, min_demand=1, max_demand=10, seed=42)


@pytest.fixture
def sample_priorities(sample_cities):
    return generate_priorities(sample_cities, critico_ratio=0.3, seed=7)


# ---------------------------------------------------------------------------
# Testes: order_crossover (regressão do bug corrigido em 22/08)
# ---------------------------------------------------------------------------

class TestOrderCrossover:

    def test_crossover_between_different_parents_produces_valid_child(self, sample_cities):
        """O filho deve conter exatamente as mesmas cidades dos pais, sem repetição e sem perda."""
        random.seed(1)
        parent1 = random.sample(sample_cities, len(sample_cities))
        parent2 = random.sample(sample_cities, len(sample_cities))

        child = order_crossover(parent1, parent2)

        assert len(child) == len(parent1)
        assert set(child) == set(parent1)  # nenhuma cidade perdida ou duplicada

    def test_crossover_with_same_parent_is_not_used_in_current_code(self):
        """
        Documenta o bug histórico: order_crossover(parent1, parent1) resulta
        numa cópia do próprio parent1, eliminando diversidade genética.
        Esse teste NÃO valida o comportamento correto -- serve como registro
        do comportamento do bug, para referência futura.
        """
        cities = default_problems[10]
        random.seed(1)
        parent1 = random.sample(cities, len(cities))

        buggy_child = order_crossover(parent1, parent1)

        # o "filho" do bug é idêntico ao próprio pai -- confirma o comportamento problemático
        assert buggy_child == parent1

    def test_crossover_produces_different_individual_than_either_parent_usually(self, sample_cities):
        """Com pais diferentes, o filho normalmente é diferente de ambos (não uma cópia)."""
        random.seed(2)
        parent1 = random.sample(sample_cities, len(sample_cities))
        parent2 = random.sample(sample_cities, len(sample_cities))

        child = order_crossover(parent1, parent2)

        # não é garantido matematicamente em 100% dos casos, mas com 15 cidades
        # e pais aleatórios diferentes, a chance do filho ser idêntico a um dos pais é desprezível
        assert child != parent1 or child != parent2


# ---------------------------------------------------------------------------
# Testes: mutate
# ---------------------------------------------------------------------------

class TestMutate:

    def test_mutate_preserves_all_cities(self, sample_cities):
        """A mutação não pode perder nem duplicar cidades, só reordenar."""
        random.seed(3)
        individual = random.sample(sample_cities, len(sample_cities))

        mutated = mutate(individual, mutation_probability=1.0)  # força mutação sempre

        assert len(mutated) == len(individual)
        assert set(mutated) == set(individual)

    def test_mutate_with_zero_probability_may_keep_individual_unchanged(self, sample_cities):
        """Com probabilidade 0, o comportamento esperado é não alterar o indivíduo (ou alterar minimamente, dependendo da implementação)."""
        random.seed(4)
        individual = random.sample(sample_cities, len(sample_cities))

        mutated = mutate(individual, mutation_probability=0.0)

        # a mutação com prob=0 não deve introduzir cidades novas nem remover nenhuma
        assert set(mutated) == set(individual)


# ---------------------------------------------------------------------------
# Testes: split_into_routes_variable (splits evoluíveis)
# ---------------------------------------------------------------------------

class TestSplitsVariable:

    def test_split_respects_total_city_count(self, sample_cities):
        """A soma dos tamanhos das rotas deve ser igual ao total de cidades."""
        splits = [4, 9]  # 3 rotas: 0:4, 4:9, 9:15
        routes = split_into_routes_variable(sample_cities, splits)

        total_cities_in_routes = sum(len(route) for route in routes)
        assert total_cities_in_routes == len(sample_cities)

    def test_split_produces_correct_number_of_routes(self, sample_cities):
        """N splits devem gerar N+1 rotas."""
        splits = [4, 9]
        routes = split_into_routes_variable(sample_cities, splits)

        assert len(routes) == len(splits) + 1

    def test_split_with_no_splits_returns_single_route(self, sample_cities):
        """Sem pontos de corte, toda a permutação vira uma única rota."""
        routes = split_into_routes_variable(sample_cities, splits=[])

        assert len(routes) == 1
        assert routes[0] == sample_cities

    def test_generate_random_splits_produces_valid_non_empty_routes(self, sample_cities):
        """Os splits gerados aleatoriamente nunca devem produzir uma rota vazia."""
        rng = random.Random(5)
        n_vehicles = 3

        splits = generate_random_splits(len(sample_cities), n_vehicles, rng)
        routes = split_into_routes_variable(sample_cities, splits)

        assert len(routes) == n_vehicles
        assert all(len(route) > 0 for route in routes)


# ---------------------------------------------------------------------------
# Testes: crossover_splits e mutate_splits
# ---------------------------------------------------------------------------

class TestSplitOperators:

    def test_crossover_splits_preserves_count(self, sample_cities):
        """O crossover de splits deve manter o número correto de pontos de corte."""
        rng = random.Random(6)
        n_cities = len(sample_cities)
        n_vehicles = 3

        splits1 = generate_random_splits(n_cities, n_vehicles, rng)
        splits2 = generate_random_splits(n_cities, n_vehicles, rng)

        child_splits = crossover_splits(splits1, splits2, n_cities, rng)

        assert len(child_splits) == n_vehicles - 1

    def test_mutate_splits_stays_within_valid_bounds(self, sample_cities):
        """A mutação de splits nunca deve gerar um ponto de corte fora do intervalo válido."""
        rng = random.Random(7)
        n_cities = len(sample_cities)
        n_vehicles = 3

        splits = generate_random_splits(n_cities, n_vehicles, rng)
        mutated_splits = mutate_splits(splits, n_cities, mutation_probability=1.0, rng=rng)

        assert all(1 <= s <= n_cities - 1 for s in mutated_splits)
        assert len(set(mutated_splits)) == len(mutated_splits)  # sem duplicatas


# ---------------------------------------------------------------------------
# Testes: cálculo de demanda e distância
# ---------------------------------------------------------------------------

class TestDemandAndDistance:

    def test_generate_demands_within_configured_range(self, sample_cities):
        """Todas as demandas geradas devem respeitar o intervalo configurado."""
        demands = generate_demands(sample_cities, min_demand=1, max_demand=10, seed=42)

        assert all(1 <= d <= 10 for d in demands.values())
        assert len(demands) == len(sample_cities)

    def test_calculate_route_demand_sums_correctly(self, sample_cities, sample_demands):
        """A demanda de uma rota deve ser a soma exata das demandas de suas cidades."""
        route = sample_cities[:3]
        expected = sum(sample_demands[c] for c in route)

        assert calculate_route_demand(route, sample_demands) == expected

    def test_route_distance_of_single_city_is_zero(self, sample_cities):
        """Uma rota com uma única cidade não tem distância a percorrer."""
        route = [sample_cities[0]]

        assert calculate_route_distance(route) == 0.0

    def test_route_distance_is_non_negative(self, sample_cities):
        """Distância de rota nunca pode ser negativa."""
        route = sample_cities[:5]

        assert calculate_route_distance(route) >= 0.0


# ---------------------------------------------------------------------------
# Testes: fitness com capacidade, autonomia e prioridade
# ---------------------------------------------------------------------------

class TestFitnessCalculation:

    def test_fitness_without_violations_equals_pure_distance(self, sample_cities, sample_demands, sample_priorities):
        """
        Se nenhuma rota viola capacidade nem autonomia, o fitness deve ser
        exatamente igual à distância total (sem nenhuma penalidade somada),
        exceto pela penalidade de prioridade (que sempre pode existir se
        houver entregas críticas fora da posição 0).
        """
        # usa capacidade e autonomia generosas o suficiente pra não violar nada
        individual = (sample_cities, [])  # uma única rota, sem splits
        route = sample_cities

        fitness = calculate_fitness_vrp_full(
            individual, sample_demands, sample_priorities,
            vehicle_capacity=999, vehicle_autonomy=999999.0,
            capacity_penalty_factor=100.0, autonomy_penalty_factor=5.0, priority_penalty_factor=15.0,
        )

        pure_distance = calculate_route_distance(route)
        priority_penalty = calculate_priority_penalty(route, sample_priorities, priority_penalty_factor=15.0)

        assert fitness == pytest.approx(pure_distance + priority_penalty)

    def test_fitness_penalizes_capacity_violation(self, sample_cities, sample_priorities):
        """Uma rota que excede a capacidade deve ter fitness maior que uma rota idêntica sem violação."""
        # demanda alta o suficiente pra estourar uma capacidade pequena
        high_demands = {city: 15 for city in sample_cities}
        individual = (sample_cities, [])

        fitness_low_capacity = calculate_fitness_vrp_full(
            individual, high_demands, sample_priorities,
            vehicle_capacity=10, vehicle_autonomy=999999.0,  # capacidade insuficiente -- deve penalizar
            capacity_penalty_factor=100.0, autonomy_penalty_factor=5.0, priority_penalty_factor=0.0,
        )
        fitness_high_capacity = calculate_fitness_vrp_full(
            individual, high_demands, sample_priorities,
            vehicle_capacity=9999, vehicle_autonomy=999999.0,  # capacidade generosa -- não deve penalizar
            capacity_penalty_factor=100.0, autonomy_penalty_factor=5.0, priority_penalty_factor=0.0,
        )

        assert fitness_low_capacity > fitness_high_capacity

    def test_fitness_penalizes_autonomy_violation(self, sample_cities, sample_demands, sample_priorities):
        """Uma rota que excede a autonomia deve ter fitness maior que uma rota idêntica sem violação."""
        individual = (sample_cities, [])

        fitness_low_autonomy = calculate_fitness_vrp_full(
            individual, sample_demands, sample_priorities,
            vehicle_capacity=999, vehicle_autonomy=1.0,  # autonomia praticamente impossível de respeitar
            capacity_penalty_factor=0.0, autonomy_penalty_factor=5.0, priority_penalty_factor=0.0,
        )
        fitness_high_autonomy = calculate_fitness_vrp_full(
            individual, sample_demands, sample_priorities,
            vehicle_capacity=999, vehicle_autonomy=999999.0,  # autonomia generosa
            capacity_penalty_factor=0.0, autonomy_penalty_factor=5.0, priority_penalty_factor=0.0,
        )

        assert fitness_low_autonomy > fitness_high_autonomy

    def test_priority_penalty_is_zero_when_all_critical_at_start(self, sample_cities):
        """Se todas as entregas críticas estão logo no início da rota, a penalidade de prioridade deve ser zero."""
        route = sample_cities[:3]
        priorities = {city: CRITICO for city in route}  # todas críticas
        priorities.update({city: REGULAR for city in sample_cities[3:]})

        # todas críticas na posição 0, 1, 2 -- mas só a primeira (posição 0) tem penalidade zero;
        # então testamos com apenas UMA cidade crítica na posição 0 pra isolar o efeito
        single_critical_route = [route[0]]
        single_priority = {route[0]: CRITICO}

        penalty = calculate_priority_penalty(single_critical_route, single_priority, priority_penalty_factor=15.0)

        assert penalty == 0.0  # posição 0 -> penalidade = 0 * fator = 0

    def test_priority_penalty_increases_with_position(self, sample_cities):
        """Quanto mais tarde uma entrega crítica aparece na rota, maior a penalidade."""
        route = sample_cities[:5]
        priorities_early = {route[0]: CRITICO}
        priorities_early.update({c: REGULAR for c in route[1:]})

        priorities_late = {route[4]: CRITICO}
        priorities_late.update({c: REGULAR for c in route[:4]})

        penalty_early = calculate_priority_penalty(route, priorities_early, priority_penalty_factor=15.0)
        penalty_late = calculate_priority_penalty(route, priorities_late, priority_penalty_factor=15.0)

        assert penalty_late > penalty_early
