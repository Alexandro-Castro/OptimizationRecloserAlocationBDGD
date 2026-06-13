from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd

from .objective import matriz_penalidade_redundancia, score_grupo


def fronteira_pareto(df: pd.DataFrame, objetivos: list[str]) -> pd.DataFrame:
    valores = df[objetivos].to_numpy()
    n = len(valores)
    pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            if np.all(valores[j] >= valores[i]) and np.any(valores[j] > valores[i]):
                pareto[i] = False
                break

    return df[pareto]


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
    if cand.empty:
        raise ValueError("Nao ha candidatos disponiveis para otimizacao.")

    if n_religadores <= 0:
        raise ValueError("n_religadores deve ser maior que zero.")

    if n_religadores > len(cand):
        raise ValueError(
            f"n_religadores={n_religadores} e maior que a quantidade de candidatos={len(cand)}."
        )

    rng = np.random.default_rng(seed)
    beneficio = cand["BENEFICIO"].to_numpy(dtype=float)
    P, HARD = matriz_penalidade_redundancia(cand, d0=d0, min_dist_serie=min_dist_serie)
    hard_penalty = 1e6

    def criar_individuo() -> np.ndarray:
        return np.sort(rng.choice(len(cand), size=n_religadores, replace=False))

    def reparar(ind: np.ndarray) -> np.ndarray:
        ind = list(dict.fromkeys(map(int, ind)))
        disponiveis = np.setdiff1d(np.arange(len(cand)), np.array(ind), assume_unique=False)

        while len(ind) < n_religadores:
            novo = int(rng.choice(disponiveis))
            ind.append(novo)
            disponiveis = np.setdiff1d(disponiveis, np.array([novo]), assume_unique=False)

        if len(ind) > n_religadores:
            ind = list(rng.choice(ind, size=n_religadores, replace=False))

        return np.sort(np.array(ind, dtype=int))

    def fitness(ind: np.ndarray) -> float:
        ind = np.array(ind, dtype=int)
        valor = beneficio[ind].sum()
        penalidade = 0.0

        for a, b in combinations(ind, 2):
            penalidade += P[a, b]
            if HARD[a, b]:
                penalidade += hard_penalty

        return float(valor - alpha_penalidade * penalidade)

    def torneio(pop: list[np.ndarray], fits: np.ndarray, k: int = 3) -> np.ndarray:
        idx = rng.choice(len(pop), size=k, replace=False)
        melhor = idx[np.argmax(fits[idx])]
        return pop[melhor]

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        pool = np.unique(np.concatenate([p1, p2]))

        if len(pool) >= n_religadores:
            filho = rng.choice(pool, size=n_religadores, replace=False)
        else:
            faltam = n_religadores - len(pool)
            disponiveis = np.setdiff1d(np.arange(len(cand)), pool, assume_unique=False)
            complemento = rng.choice(disponiveis, size=faltam, replace=False)
            filho = np.concatenate([pool, complemento])

        return reparar(filho)

    def mutar(ind: np.ndarray) -> np.ndarray:
        ind = ind.copy()

        if rng.random() < taxa_mutacao:
            pos = rng.integers(0, n_religadores)
            disponiveis = np.setdiff1d(np.arange(len(cand)), ind, assume_unique=False)

            if len(disponiveis) > 0:
                ind[pos] = int(rng.choice(disponiveis))

        return reparar(ind)

    pop = [criar_individuo() for _ in range(pop_size)]
    melhor_ind = None
    melhor_fit = -np.inf
    historico = []

    for g in range(geracoes):
        fits = np.array([fitness(ind) for ind in pop])
        idx_melhor = int(np.argmax(fits))

        if fits[idx_melhor] > melhor_fit:
            melhor_fit = float(fits[idx_melhor])
            melhor_ind = pop[idx_melhor].copy()

        historico.append(
            {
                "geracao": g,
                "melhor_fitness": melhor_fit,
                "media_fitness": float(np.mean(fits)),
            }
        )

        elite_idx = np.argsort(fits)[-elite:]
        nova_pop = [pop[i].copy() for i in elite_idx]

        while len(nova_pop) < pop_size:
            p1 = torneio(pop, fits)
            p2 = torneio(pop, fits)
            filho = crossover(p1, p2)
            filho = mutar(filho)
            nova_pop.append(filho)

        pop = nova_pop

    if melhor_ind is None:
        raise RuntimeError("O algoritmo genetico nao produziu solucao.")

    solucao = cand.iloc[melhor_ind].copy().reset_index(drop=True)
    solucao["SELECIONADO"] = 1

    penalidade_total = 0.0
    pares_redundantes = []
    for a, b in combinations(melhor_ind, 2):
        penalidade_total += P[a, b]

        if P[a, b] > 0:
            pares_redundantes.append(
                {
                    "PAC_i": cand.iloc[a]["PAC"],
                    "PAC_j": cand.iloc[b]["PAC"],
                    "PENALIDADE_PAR": P[a, b],
                    "HARD": bool(HARD[a, b]),
                    "DIST_i": cand.iloc[a]["DIST_RAIZ"],
                    "DIST_j": cand.iloc[b]["DIST_RAIZ"],
                    "UCs_JUS_i": cand.iloc[a]["UCs_JUS"],
                    "UCs_JUS_j": cand.iloc[b]["UCs_JUS"],
                }
            )

    info = {
        "objetivo": melhor_fit,
        "beneficio_total": float(solucao["BENEFICIO"].sum()),
        "penalidade_total": float(penalidade_total),
        "n_candidatos": len(cand),
        "n_religadores": n_religadores,
        "historico": pd.DataFrame(historico),
        "pares_redundantes": pd.DataFrame(pares_redundantes),
    }
    return solucao, info
