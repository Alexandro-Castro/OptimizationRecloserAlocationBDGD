from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .graph_builder import distancia_topologica


def eh_ancestral(tin_a: int, tout_a: int, tin_b: int) -> bool:
    return tin_a <= tin_b < tout_a


def matriz_penalidade_redundancia(
    cand: pd.DataFrame,
    d0: float = 1000.0,
    peso_sobreposicao: float = 0.70,
    peso_proximidade: float = 0.30,
    min_dist_serie: float = 500.0,
    jaccard_hard: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(cand)
    P = np.zeros((n, n), dtype=float)
    HARD = np.zeros((n, n), dtype=bool)

    tin = cand["TIN"].to_numpy()
    tout = cand["TOUT"].to_numpy()
    dist = cand["DIST_RAIZ"].to_numpy()
    W = cand["UCs_JUS"].clip(lower=1.0).to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            i_anc_j = eh_ancestral(tin[i], tout[i], tin[j])
            j_anc_i = eh_ancestral(tin[j], tout[j], tin[i])

            if not i_anc_j and not j_anc_i:
                continue

            if i_anc_j:
                up, down = i, j
            else:
                up, down = j, i

            jaccard = W[down] / max(W[up], 1e-9)
            distancia_serie = abs(dist[i] - dist[j])
            proximidade = np.exp(-distancia_serie / max(d0, 1e-9))
            penalidade = peso_sobreposicao * jaccard + peso_proximidade * proximidade

            P[i, j] = penalidade
            P[j, i] = penalidade

            if distancia_serie < min_dist_serie and jaccard >= jaccard_hard:
                HARD[i, j] = True
                HARD[j, i] = True

    return P, HARD


def penalizacao_proximidade(df_sol: pd.DataFrame, grafo: dict[str, set[str]]) -> float:
    penalizacao = 0.0
    pacs = df_sol["PAC_1"].astype(str).tolist()

    for pac1, pac2 in combinations(pacs, 2):
        dist = distancia_topologica(grafo, pac1, pac2, limite=50)
        if dist <= 50:
            penalizacao += 51 - dist

    return penalizacao


def score_grupo(df_sol: pd.DataFrame, grafo: dict[str, set[str]]) -> float:
    score_individual = (
        0.55 * df_sol["UCs"]
        + 0.10 * df_sol["DEC"]
        + 0.10 * df_sol["FEC"]
        + 0.25 * df_sol["DIST_RAIZ"]
    )
    balanceamento = df_sol[["UCs", "DEC", "FEC", "DIST_RAIZ"]].min(axis=1)
    score_base = (score_individual * balanceamento).sum()
    penalizacao = penalizacao_proximidade(df_sol, grafo)
    return float(score_base - 0.5 * penalizacao)
