from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .objective import evaluate_solution, matriz_penalidade_redundancia, score_grupo


def simulated_annealing(
    df_pareto: pd.DataFrame,
    grafo: dict[str, set[str]],
    n_chaves: int = 5,
    T0: float = 100,
    Tf: float = 0.001,
    alpha: float = 0.95,
    iter_por_temp: int = 100,
) -> tuple[pd.DataFrame, float]:
    df_pareto = df_pareto.reset_index(drop=True)
    indices = np.arange(len(df_pareto))
    solucao_atual = random.sample(list(indices), n_chaves)
    score_atual = score_grupo(df_pareto.loc[solucao_atual], grafo)

    melhor_solucao = solucao_atual.copy()
    melhor_score = score_atual
    T = T0

    while T > Tf:
        for _ in range(iter_por_temp):
            vizinho = solucao_atual.copy()
            posicao = random.randint(0, n_chaves - 1)
            disponiveis = list(set(indices) - set(vizinho))
            novo_indice = random.choice(disponiveis)
            vizinho[posicao] = novo_indice
            score_vizinho = score_grupo(df_pareto.loc[vizinho], grafo)
            delta = score_vizinho - score_atual

            if delta > 0:
                aceita = True
            else:
                aceita = np.random.rand() < np.exp(delta / T)

            if aceita:
                solucao_atual = vizinho
                score_atual = score_vizinho

                if score_atual > melhor_score:
                    melhor_score = score_atual
                    melhor_solucao = vizinho.copy()

        T *= alpha

    return df_pareto.loc[melhor_solucao], float(melhor_score)


def _candidate_ids(candidates_df: pd.DataFrame) -> list[int]:
    if "ID_CAND" in candidates_df.columns:
        return [int(candidate_id) for candidate_id in candidates_df["ID_CAND"].tolist()]
    return list(range(len(candidates_df)))


def _solution_dataframe(candidates_df: pd.DataFrame, selected_ids: list[int]) -> pd.DataFrame:
    if "ID_CAND" in candidates_df.columns:
        selecionados = candidates_df[candidates_df["ID_CAND"].astype(int).isin(selected_ids)].copy()
    else:
        selecionados = candidates_df.iloc[selected_ids].copy()
        selecionados["ID_CAND"] = selected_ids

    colunas = [
        "ID_CAND",
        "PAC",
        "PAI",
        "DIST_RAIZ",
        "UCs_JUS",
        "DIC_JUS",
        "FIC_JUS",
        "BENEFICIO",
    ]
    colunas_existentes = [col for col in colunas if col in selecionados.columns]
    solucao = selecionados[colunas_existentes].copy().reset_index(drop=True)
    solucao["SELECIONADO"] = 1
    return solucao


def genetic_algorithm_reclosers(
    candidates_df: pd.DataFrame,
    n_religadores: int,
    population_size: int = 120,
    generations: int = 250,
    mutation_rate: float = 0.25,
    elite_size: int = 8,
    random_seed: int = 42,
    alpha_penalty: float = 1.0,
    d0: float = 1000.0,
    min_dist_serie: float = 500.0,
) -> dict[str, object]:
    if candidates_df.empty:
        raise ValueError("Nao ha candidatos disponiveis para otimizacao.")

    if n_religadores <= 0:
        raise ValueError("n_religadores deve ser maior que zero.")

    candidate_ids = _candidate_ids(candidates_df)
    if n_religadores > len(candidate_ids):
        raise ValueError(
            f"n_religadores={n_religadores} e maior que a quantidade de candidatos={len(candidate_ids)}."
        )

    if population_size <= 0:
        raise ValueError("population_size deve ser maior que zero.")

    if generations <= 0:
        raise ValueError("generations deve ser maior que zero.")

    elite_size = max(0, min(int(elite_size), int(population_size)))
    rng = np.random.default_rng(random_seed)
    penalty_matrix, _ = matriz_penalidade_redundancia(
        candidates_df,
        d0=d0,
        min_dist_serie=min_dist_serie,
    )

    def reparar(individuo: list[int] | np.ndarray) -> list[int]:
        genes = list(dict.fromkeys(int(gene) for gene in individuo))
        disponiveis = [candidate_id for candidate_id in candidate_ids if candidate_id not in genes]

        while len(genes) < n_religadores:
            novo = int(rng.choice(disponiveis))
            genes.append(novo)
            disponiveis.remove(novo)

        if len(genes) > n_religadores:
            genes = [int(gene) for gene in rng.choice(genes, size=n_religadores, replace=False)]

        return sorted(genes)

    def criar_individuo() -> list[int]:
        return sorted(int(gene) for gene in rng.choice(candidate_ids, size=n_religadores, replace=False))

    def avaliar(individuo: list[int]) -> dict[str, object]:
        return evaluate_solution(individuo, candidates_df, penalty_matrix, alpha_penalty)

    def fitness(individuo: list[int]) -> float:
        return float(avaliar(individuo)["objetivo_total"])

    def torneio(populacao: list[list[int]], fits: np.ndarray, k: int = 3) -> list[int]:
        k = min(k, len(populacao))
        idx = rng.choice(len(populacao), size=k, replace=False)
        melhor = int(idx[np.argmax(fits[idx])])
        return populacao[melhor].copy()

    def crossover(pai_1: list[int], pai_2: list[int]) -> list[int]:
        pool = sorted(set(pai_1).union(pai_2))

        if len(pool) >= n_religadores:
            filho = [int(gene) for gene in rng.choice(pool, size=n_religadores, replace=False)]
        else:
            faltam = n_religadores - len(pool)
            disponiveis = [candidate_id for candidate_id in candidate_ids if candidate_id not in pool]
            complemento = [int(gene) for gene in rng.choice(disponiveis, size=faltam, replace=False)]
            filho = pool + complemento

        return reparar(filho)

    def mutar(individuo: list[int]) -> list[int]:
        mutado = individuo.copy()

        if rng.random() < mutation_rate:
            disponiveis = [candidate_id for candidate_id in candidate_ids if candidate_id not in mutado]

            if disponiveis:
                pos = int(rng.integers(0, n_religadores))
                mutado[pos] = int(rng.choice(disponiveis))

        return reparar(mutado)

    populacao = [criar_individuo() for _ in range(population_size)]
    melhor_individuo: list[int] | None = None
    melhor_avaliacao: dict[str, object] | None = None
    historico = []

    for geracao in range(generations):
        fits = np.array([fitness(individuo) for individuo in populacao], dtype=float)
        idx_melhor = int(np.argmax(fits))
        avaliacao_geracao = avaliar(populacao[idx_melhor])

        if melhor_avaliacao is None or float(avaliacao_geracao["objetivo_total"]) > float(
            melhor_avaliacao["objetivo_total"]
        ):
            melhor_individuo = populacao[idx_melhor].copy()
            melhor_avaliacao = avaliacao_geracao

        historico.append(
            {
                "geracao": geracao,
                "melhor_objetivo": float(melhor_avaliacao["objetivo_total"]),
                "media_objetivo": float(np.mean(fits)),
                "melhor_beneficio": float(melhor_avaliacao["beneficio_total"]),
                "melhor_penalidade": float(melhor_avaliacao["penalidade_total"]),
            }
        )

        elite_idx = np.argsort(fits)[-elite_size:] if elite_size > 0 else np.array([], dtype=int)
        nova_populacao = [populacao[int(i)].copy() for i in elite_idx]

        while len(nova_populacao) < population_size:
            pai_1 = torneio(populacao, fits)
            pai_2 = torneio(populacao, fits)
            filho = crossover(pai_1, pai_2)
            filho = mutar(filho)
            nova_populacao.append(filho)

        populacao = nova_populacao

    if melhor_individuo is None or melhor_avaliacao is None:
        raise RuntimeError("O algoritmo genetico nao produziu solucao.")

    solucao_df = _solution_dataframe(candidates_df, melhor_individuo)
    historico_df = pd.DataFrame(historico)
    pares_penalizados_df = pd.DataFrame(melhor_avaliacao["pares_penalizados"])

    return {
        "melhor_solucao": solucao_df,
        "solution_df": solucao_df,
        "selected_candidate_ids": sorted(melhor_individuo),
        "historico": historico_df,
        "beneficio_total": float(melhor_avaliacao["beneficio_total"]),
        "penalidade_total": float(melhor_avaliacao["penalidade_total"]),
        "objetivo_total": float(melhor_avaliacao["objetivo_total"]),
        "pares_penalizados": pares_penalizados_df,
        "penalty_matrix": penalty_matrix,
    }


def otimizar_religadores_ga(
    cand: pd.DataFrame,
    n_religadores: int,
    alpha_penalidade: float = 1.0,
    d0: float = 1000.0,
    min_dist_serie: float = 500.0,
    pop_size: int = 120,
    geracoes: int = 250,
    taxa_mutacao: float = 0.25,
    elite: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resultado = genetic_algorithm_reclosers(
        cand,
        n_religadores=n_religadores,
        population_size=pop_size,
        generations=geracoes,
        mutation_rate=taxa_mutacao,
        elite_size=elite,
        random_seed=seed,
        alpha_penalty=alpha_penalidade,
        d0=d0,
        min_dist_serie=min_dist_serie,
    )
    info = {
        "objetivo": resultado["objetivo_total"],
        "objetivo_total": resultado["objetivo_total"],
        "beneficio_total": resultado["beneficio_total"],
        "penalidade_total": resultado["penalidade_total"],
        "n_candidatos": len(cand),
        "n_religadores": n_religadores,
        "selected_candidate_ids": resultado["selected_candidate_ids"],
        "historico": resultado["historico"],
        "pares_redundantes": resultado["pares_penalizados"],
    }
    return resultado["solution_df"], info
