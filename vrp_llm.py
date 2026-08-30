"""
Integração com LLM (Google Gemini API) para o VRP -- Tech Challenge Fase 2.

Cobre os 3 requisitos do enunciado (item 3 -- Integração com LLMs):
1. Gerar instruções detalhadas para motoristas/equipes de entrega
2. Gerar relatórios de eficiência das rotas
3. Responder perguntas em linguagem natural sobre rotas e entregas

Provedor: Google Gemini API (via Google AI Studio) -- free tier permanente,
sem cartão de crédito, sem prazo de expiração, apenas conta Google. Trocado
a partir de tentativas anteriores com Claude (Anthropic, requer crédito
pago após o trial inicial) e Groq (problema de login no momento do
desenvolvimento).
Modelo usado: gemini-3.6-flash (elegível para a free tier).

IMPORTANTE: este arquivo precisa da variável de ambiente GEMINI_API_KEY
configurada para funcionar. Sem ela, roda em modo "dry-run" (mostra os
prompts que seriam enviados, sem chamar a API de verdade) -- útil para
revisar o texto dos prompts antes de fazer chamadas reais.

Para configurar a chave (no terminal, antes de rodar o script):
    export GEMINI_API_KEY="sua-chave-aqui"        # Mac/Linux
    set GEMINI_API_KEY=sua-chave-aqui              # Windows (cmd)
    $env:GEMINI_API_KEY="sua-chave-aqui"            # Windows (PowerShell)

Chave gratuita disponível em: https://aistudio.google.com/apikey (basta uma
conta Google, sem cartão de crédito)
"""

import os
from typing import List, Tuple, Dict

MODEL = "gemini-3.6-flash"


def get_client():
    """
    Cria o client do Gemini. Retorna None se a biblioteca ou a chave de
    API não estiverem disponíveis -- nesse caso, o script roda em modo
    dry-run (mostra os prompts sem enviar).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        return None


def call_llm(client, prompt: str, max_tokens: int = 1024) -> str:
    """
    Envia um prompt para o Gemini e retorna o texto da resposta.
    Se o client for None (modo dry-run), apenas devolve o prompt formatado
    para revisão, sem chamar a API.
    """
    if client is None:
        return f"[DRY-RUN -- prompt que seria enviado à API]\n\n{prompt}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text


def format_route_for_prompt(route: List[Tuple[float, float]], priorities: Dict[Tuple[float, float], str], demands: Dict[Tuple[float, float], int], vehicle_number: int) -> str:
    """Formata os dados de uma rota em texto legível para incluir no prompt."""
    lines = [f"Veículo {vehicle_number} -- {len(route)} paradas:"]
    for i, city in enumerate(route):
        priority = priorities.get(city, "regular")
        demand = demands.get(city, 0)
        tag = "[CRÍTICO]" if priority == "critico" else "[regular]"
        lines.append(f"  Parada {i+1}: coordenadas {city} -- {tag} -- carga: {demand} unidades")
    return "\n".join(lines)


def generate_driver_instructions(client, route: List[Tuple[float, float]], priorities: Dict[Tuple[float, float], str], demands: Dict[Tuple[float, float], int], vehicle_number: int) -> str:
    """
    Gera instruções em linguagem natural para o motorista responsável por
    uma rota específica -- requisito do enunciado.
    """
    route_description = format_route_for_prompt(route, priorities, demands, vehicle_number)

    prompt = f"""Você é um assistente de logística hospitalar. Gere instruções claras e objetivas
para o motorista do veículo {vehicle_number}, que vai fazer a seguinte rota de entrega de
medicamentos e insumos:

{route_description}

As instruções devem:
- Confirmar a ordem das paradas
- Destacar quais entregas são CRÍTICAS (medicamentos urgentes) e reforçar que devem ser
  priorizadas e entregues o mais rápido possível
- Ser objetivas, em tom profissional, adequadas para leitura rápida por um motorista antes de
  sair para a rota
- Ter no máximo 150 palavras"""

    return call_llm(client, prompt, max_tokens=400)


def generate_efficiency_report(client, all_routes: List[List[Tuple[float, float]]], priorities: Dict[Tuple[float, float], str], demands: Dict[Tuple[float, float], int], route_distances: List[float], vehicle_capacity: int, vehicle_autonomy: float) -> str:
    """
    Gera um relatório de eficiência agregando todas as rotas -- requisito do
    enunciado ("relatórios diários/semanais sobre eficiência de rotas,
    economia de tempo e recursos").
    """
    summary_lines = []
    total_distance = 0.0
    total_demand = 0
    total_criticas = 0

    for i, route in enumerate(all_routes):
        dist = route_distances[i]
        demand = sum(demands.get(c, 0) for c in route)
        criticas = sum(1 for c in route if priorities.get(c) == "critico")
        total_distance += dist
        total_demand += demand
        total_criticas += criticas
        summary_lines.append(
            f"Veículo {i+1}: {len(route)} paradas, {dist:.1f} de distância, "
            f"{demand}/{vehicle_capacity} de carga, {criticas} entregas críticas"
        )

    summary = "\n".join(summary_lines)

    prompt = f"""Você é um analista de logística hospitalar. Com base nos dados abaixo de uma
rodada de otimização de rotas (algoritmo genético), escreva um relatório breve de eficiência,
destinado à gestão do hospital.

Dados da rodada:
- Número de veículos: {len(all_routes)}
- Capacidade máxima por veículo: {vehicle_capacity} unidades
- Autonomia máxima por veículo: {vehicle_autonomy} (mesma unidade das coordenadas)
- Distância total percorrida pela frota: {total_distance:.1f}
- Carga total transportada: {total_demand} unidades
- Total de entregas críticas: {total_criticas}

Detalhe por veículo:
{summary}

O relatório deve:
- Resumir o desempenho geral da rodada (distância total, uso de capacidade)
- Comentar se as entregas críticas parecem bem distribuídas/priorizadas
- Sugerir, em 1-2 frases, algo que poderia ser melhorado na próxima rodada
- Ter tom profissional, objetivo, no máximo 200 palavras"""

    return call_llm(client, prompt, max_tokens=500)


def answer_question_about_routes(client, question: str, all_routes: List[List[Tuple[float, float]]], priorities: Dict[Tuple[float, float], str], demands: Dict[Tuple[float, float], int], route_distances: List[float]) -> str:
    """
    Responde a uma pergunta em linguagem natural sobre as rotas -- requisito
    do enunciado ("permitir que o sistema responda a perguntas em linguagem
    natural sobre as rotas e entregas").
    """
    context_lines = []
    for i, route in enumerate(all_routes):
        route_desc = format_route_for_prompt(route, priorities, demands, i + 1)
        context_lines.append(f"{route_desc}\n  Distância total da rota: {route_distances[i]:.1f}")

    context = "\n\n".join(context_lines)

    prompt = f"""Você é um assistente que responde perguntas sobre as rotas de entrega de
medicamentos e insumos de um hospital, otimizadas por algoritmo genético.

Dados completos das rotas atuais:

{context}

Pergunta do usuário: {question}

Responda de forma direta e objetiva, usando apenas os dados fornecidos acima. Se a pergunta não
puder ser respondida com esses dados, diga isso claramente."""

    return call_llm(client, prompt, max_tokens=400)


if __name__ == '__main__':
    # Exemplo de uso com dados fictícios simples (não depende do vrp_capacity_autonomy_priority.py
    # para este teste isolado -- em produção, os dados viriam da melhor solução do GA)
    example_routes = [
        [(512, 317), (741, 72), (552, 50)],
        [(772, 346), (637, 12)],
    ]
    example_priorities = {
        (512, 317): "critico", (741, 72): "regular", (552, 50): "critico",
        (772, 346): "regular", (637, 12): "critico",
    }
    example_demands = {
        (512, 317): 5, (741, 72): 3, (552, 50): 8,
        (772, 346): 4, (637, 12): 6,
    }
    example_distances = [320.5, 410.2]

    client = get_client()
    if client is None:
        print("=" * 70)
        print("MODO DRY-RUN: variável GEMINI_API_KEY não encontrada.")
        print("Os prompts abaixo seriam enviados à API, mas não estão sendo enviados.")
        print("Configure a chave (gratuita em aistudio.google.com/apikey) para testar com respostas reais.")
        print("=" * 70)
        print()

    print("--- Instruções para o motorista do Veículo 1 ---")
    print(generate_driver_instructions(client, example_routes[0], example_priorities, example_demands, 1))
    print()

    print("--- Relatório de eficiência ---")
    print(generate_efficiency_report(
        client, example_routes, example_priorities, example_demands, example_distances,
        vehicle_capacity=20, vehicle_autonomy=450.0,
    ))
    print()

    print("--- Pergunta em linguagem natural ---")
    pergunta = "Quantas entregas críticas existem no total e em quais veículos elas estão?"
    print(f"Pergunta: {pergunta}\n")
    print(answer_question_about_routes(client, pergunta, example_routes, example_priorities, example_demands, example_distances))
