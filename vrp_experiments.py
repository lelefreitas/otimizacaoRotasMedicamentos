"""
Experimentos comparativos: roda o algoritmo genético com diferentes
configurações (tamanho de população e taxa de mutação) e compara os
resultados -- fitness final, velocidade de convergência e tempo de
execução.

Atende ao requisito do enunciado de comparativo de desempenho, permitindo
justificar no relatório técnico quais parâmetros funcionam melhor para
este problema específico.

Rodar com: python vrp_experiments.py
"""

import time
import matplotlib.pyplot as plt

from vrp_pipeline import run_genetic_algorithm


# ---------------------------------------------------------------------------
# Configurações a comparar
# ---------------------------------------------------------------------------
# Mantemos os demais parâmetros fixos (mesmos já validados em execuções
# anteriores) e variamos apenas população e taxa de mutação, para isolar
# o efeito de cada uma.

BASE_PARAMS = dict(
    n_cities=15,
    n_vehicles=3,
    split_mutation_probability=0.3,
    vehicle_capacity=20,
    vehicle_autonomy=450.0,
    capacity_penalty_factor=100.0,
    autonomy_penalty_factor=5.0,
    priority_penalty_factor=15.0,
    demand_seed=42,
    priority_seed=7,
    critico_ratio=0.3,
    rng_seed=1,  # mesma seed em todos os experimentos, para comparação justa
)

EXPERIMENTS = [
    {"name": "Baseline (pop=100, mut=0.3)", "population_size": 100, "mutation_probability": 0.3, "n_generations": 150},
    {"name": "População pequena (pop=30, mut=0.3)", "population_size": 30, "mutation_probability": 0.3, "n_generations": 150},
    {"name": "População grande (pop=250, mut=0.3)", "population_size": 250, "mutation_probability": 0.3, "n_generations": 150},
    {"name": "Mutação baixa (pop=100, mut=0.1)", "population_size": 100, "mutation_probability": 0.1, "n_generations": 150},
    {"name": "Mutação alta (pop=100, mut=0.6)", "population_size": 100, "mutation_probability": 0.6, "n_generations": 150},
]


def run_experiment(config: dict):
    """Roda uma configuração específica e retorna as métricas coletadas."""
    start_time = time.time()

    best_solution, best_fitness, demands, priorities, fitness_history = run_genetic_algorithm(
        n_cities=BASE_PARAMS["n_cities"],
        n_vehicles=BASE_PARAMS["n_vehicles"],
        population_size=config["population_size"],
        n_generations=config["n_generations"],
        mutation_probability=config["mutation_probability"],
        split_mutation_probability=BASE_PARAMS["split_mutation_probability"],
        vehicle_capacity=BASE_PARAMS["vehicle_capacity"],
        vehicle_autonomy=BASE_PARAMS["vehicle_autonomy"],
        capacity_penalty_factor=BASE_PARAMS["capacity_penalty_factor"],
        autonomy_penalty_factor=BASE_PARAMS["autonomy_penalty_factor"],
        priority_penalty_factor=BASE_PARAMS["priority_penalty_factor"],
        demand_seed=BASE_PARAMS["demand_seed"],
        priority_seed=BASE_PARAMS["priority_seed"],
        critico_ratio=BASE_PARAMS["critico_ratio"],
        rng_seed=BASE_PARAMS["rng_seed"],
        verbose=False,  # silencia o log gera-a-geração para não poluir a saída do comparativo
        track_history=True,
    )

    elapsed = time.time() - start_time

    # geração em que o fitness chegou a 105% do valor final (proxy de "convergiu o suficiente")
    convergence_threshold = best_fitness * 1.05
    generation_converged = next(
        (i for i, f in enumerate(fitness_history) if f <= convergence_threshold),
        len(fitness_history) - 1,
    )

    return {
        "name": config["name"],
        "population_size": config["population_size"],
        "mutation_probability": config["mutation_probability"],
        "final_fitness": best_fitness,
        "generation_converged": generation_converged,
        "elapsed_seconds": elapsed,
        "fitness_history": fitness_history,
    }


if __name__ == '__main__':
    print("=" * 90)
    print("COMPARATIVO DE DESEMPENHO -- Diferentes Configurações do Algoritmo Genético")
    print("=" * 90)
    print(f"\nRodando {len(EXPERIMENTS)} experimentos, cada um com {EXPERIMENTS[0]['n_generations']} gerações...\n")

    results = []
    for config in EXPERIMENTS:
        print(f"Rodando: {config['name']}...")
        result = run_experiment(config)
        results.append(result)
        print(f"  -> fitness final = {result['final_fitness']:.2f} | "
              f"convergiu na geração {result['generation_converged']} | "
              f"tempo = {result['elapsed_seconds']:.2f}s\n")

    # -- Tabela-resumo --
    print("\n" + "=" * 90)
    print("RESUMO COMPARATIVO")
    print("=" * 90)
    print(f"{'Configuração':<40} {'Fitness final':>15} {'Geração conv.':>15} {'Tempo (s)':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<40} {r['final_fitness']:>15.2f} {r['generation_converged']:>15} {r['elapsed_seconds']:>12.2f}")

    best_result = min(results, key=lambda r: r["final_fitness"])
    fastest_result = min(results, key=lambda r: r["generation_converged"])
    print(f"\nMelhor fitness final: {best_result['name']} ({best_result['final_fitness']:.2f})")
    print(f"Convergência mais rápida: {fastest_result['name']} (geração {fastest_result['generation_converged']})")

    # -- Gráfico comparativo das curvas de convergência --
    fig, ax = plt.subplots(figsize=(11, 7))
    for r in results:
        ax.plot(r["fitness_history"], label=r["name"], linewidth=2)

    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness (distância + penalidades, menor é melhor)")
    ax.set_title("Comparativo de Convergência -- Diferentes Configurações do GA", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparativo_configuracoes_ga.png", dpi=150, bbox_inches="tight")
    print("\nGráfico comparativo salvo em: comparativo_configuracoes_ga.png")
    plt.close(fig)
