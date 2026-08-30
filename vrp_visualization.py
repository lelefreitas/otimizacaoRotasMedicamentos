"""
Visualização das rotas otimizadas em um mapa (requisito do enunciado:
"Visualizar as rotas otimizadas em um mapa para fácil interpretação").

Usa matplotlib para desenhar um mapa 2D das rotas: cada veículo recebe uma
cor distinta, as paradas são conectadas em ordem (mostrando o caminho
percorrido), e entregas críticas são destacadas visualmente (marcador
diferente + contorno vermelho) para facilitar a leitura rápida por parte
da equipe de logística.

As coordenadas do problema são abstratas (não são latitude/longitude reais
de um mapa geográfico) -- por isso a visualização usa um plano cartesiano
simples, análogo ao que já era usado no Pygame do código-base, mas com
apresentação mais polida e fácil de exportar como imagem para relatórios.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict

VEHICLE_COLORS = [
    "#1f77b4",  # azul
    "#ff7f0e",  # laranja
    "#2ca02c",  # verde
    "#d62728",  # vermelho
    "#9467bd",  # roxo
    "#8c564b",  # marrom
]


def plot_routes(
    routes: List[List[Tuple[float, float]]],
    priorities: Dict[Tuple[float, float], str],
    demands: Dict[Tuple[float, float], int],
    title: str = "Rotas Otimizadas de Entrega de Medicamentos e Insumos",
    save_path: str = None,
    show: bool = True,
):
    """
    Desenha um mapa com todas as rotas da frota.

    Parâmetros:
        routes: lista de rotas, cada rota é uma lista de coordenadas (cidades) em ordem de visita
        priorities: dicionário cidade -> "critico" ou "regular"
        demands: dicionário cidade -> quantidade de carga
        title: título do gráfico
        save_path: se fornecido, salva a imagem nesse caminho (ex: "rotas.png")
        show: se True, exibe o gráfico numa janela (use False em ambientes sem display)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, route in enumerate(routes):
        color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]

        xs = [city[0] for city in route]
        ys = [city[1] for city in route]

        # linha conectando as paradas na ordem da rota
        ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.6, zorder=1,
                label=f"Veículo {i + 1} ({len(route)} paradas)")

        # seta indicando o sentido do percurso entre cada par de paradas consecutivas
        for j in range(len(route) - 1):
            x1, y1 = route[j]
            x2, y2 = route[j + 1]
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=1.5),
                zorder=1,
            )

        # marcadores das paradas: crítica (estrela vermelha) vs regular (círculo)
        for position, city in enumerate(route):
            is_critical = priorities.get(city) == "critico"
            demand = demands.get(city, 0)

            if is_critical:
                ax.scatter(*city, marker="*", s=300, color=color, edgecolors="red",
                           linewidths=2, zorder=3)
            else:
                ax.scatter(*city, marker="o", s=150, color=color, edgecolors="black",
                           linewidths=1, zorder=2)

            # número da ordem de visita + carga, ao lado de cada parada
            ax.annotate(
                f"{position}\n({demand}u)", city,
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="black",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")

    # legenda de veículos (precisa ser guardada com add_artist, senão a
    # segunda chamada de ax.legend() abaixo a substitui em vez de coexistir)
    vehicle_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.add_artist(vehicle_legend)

    # legenda extra explicando os símbolos (crítico vs regular)
    critical_patch = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                                  markeredgecolor="red", markersize=15, label="Entrega crítica")
    regular_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                                 markeredgecolor="black", markersize=10, label="Entrega regular")
    ax.legend(handles=[critical_patch, regular_patch], loc="lower left",
              bbox_to_anchor=(1.02, 0.0), fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Mapa salvo em: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    # Exemplo standalone com dados fictícios, para testar a visualização isoladamente
    example_routes = [
        [(512, 317), (741, 72), (552, 50)],
        [(772, 346), (637, 12), (400, 200)],
    ]
    example_priorities = {
        (512, 317): "critico", (741, 72): "regular", (552, 50): "critico",
        (772, 346): "regular", (637, 12): "critico", (400, 200): "regular",
    }
    example_demands = {
        (512, 317): 5, (741, 72): 3, (552, 50): 8,
        (772, 346): 4, (637, 12): 6, (400, 200): 2,
    }

    plot_routes(
        example_routes, example_priorities, example_demands,
        title="Exemplo -- Rotas de Entrega (dados fictícios)",
        save_path="exemplo_mapa_rotas.png",
        show=False,  # False para rodar em ambientes sem display gráfico
    )
