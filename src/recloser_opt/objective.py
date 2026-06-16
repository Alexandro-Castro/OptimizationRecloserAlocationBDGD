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
    ucs_jus = cand["UCs_JUS"].clip(lower=0.0).to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            i_anc_j = eh_ancestral(tin[i], tout[i], tin[j])
            j_anc_i = eh_ancestral(tin[j], tout[j], tin[i])

            if not i_anc_j and not j_anc_i:
                continue

            max_ucs = max(ucs_jus[i], ucs_jus[j], 1e-9)
            sobreposicao = min(ucs_jus[i], ucs_jus[j]) / max_ucs
            distancia_serie = abs(float(dist[i]) - float(dist[j]))
            proximidade = np.exp(-distancia_serie / max(d0, 1e-9))
            penalidade = peso_sobreposicao * sobreposicao + peso_proximidade * proximidade

            P[i, j] = penalidade
            P[j, i] = penalidade

            if distancia_serie < min_dist_serie and sobreposicao >= jaccard_hard:
                HARD[i, j] = True
                HARD[j, i] = True

    return P, HARD


def _candidate_ids_to_positions(
    selected_candidate_ids: list[int] | tuple[int, ...] | np.ndarray,
    candidates_df: pd.DataFrame,
) -> list[int]:
    selected = [int(candidate_id) for candidate_id in selected_candidate_ids]

    if len(selected) != len(set(selected)):
        raise ValueError("selected_candidate_ids nao pode conter candidatos repetidos.")

    if "ID_CAND" not in candidates_df.columns:
        max_pos = len(candidates_df) - 1
        invalidos = [idx for idx in selected if idx < 0 or idx > max_pos]
        if invalidos:
            raise ValueError(f"Indices de candidatos invalidos: {invalidos}")
        return selected

    pos_por_id = {
        int(candidate_id): pos
        for pos, candidate_id in enumerate(candidates_df["ID_CAND"].astype(int).tolist())
    }
    invalidos = [candidate_id for candidate_id in selected if candidate_id not in pos_por_id]
    if invalidos:
        raise ValueError(f"ID_CAND inexistente em candidates_df: {invalidos}")

    return [pos_por_id[candidate_id] for candidate_id in selected]


def _beneficios_por_candidato(candidates_df: pd.DataFrame) -> pd.Series:
    colunas = ["DIC_JUS_N", "FIC_JUS_N", "UCs_JUS_N", "TRONCO_AUTO"]
    faltantes = [col for col in colunas if col not in candidates_df.columns]
    if faltantes and "BENEFICIO" in candidates_df.columns:
        return pd.to_numeric(candidates_df["BENEFICIO"], errors="coerce").fillna(0.0)

    if faltantes:
        raise ValueError(f"Colunas ausentes para calcular BENEFICIO: {faltantes}")

    return (
        pd.to_numeric(candidates_df["DIC_JUS_N"], errors="coerce").fillna(0.0)
        / pd.to_numeric(candidates_df["UCs_JUS_N"], errors="coerce").fillna(0.0)
    )


def evaluate_solution(
    selected_candidate_ids: list[int] | tuple[int, ...] | np.ndarray,
    candidates_df: pd.DataFrame,
    penalty_matrix: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    positions = _candidate_ids_to_positions(selected_candidate_ids, candidates_df)
    beneficios = _beneficios_por_candidato(candidates_df).to_numpy(dtype=float)

    if penalty_matrix.shape[0] != len(candidates_df) or penalty_matrix.shape[1] != len(candidates_df):
        raise ValueError("penalty_matrix deve ter dimensoes iguais ao numero de candidatos.")

    beneficio_total = float(beneficios[positions].sum())
    penalidade_total = 0.0
    pares_penalizados = []

    for pos_i, pos_j in combinations(positions, 2):
        penalidade = float(penalty_matrix[pos_i, pos_j])
        if penalidade <= 0:
            continue

        penalidade_total += penalidade
        cand_i = candidates_df.iloc[pos_i]
        cand_j = candidates_df.iloc[pos_j]
        pares_penalizados.append(
            {
                "ID_CAND_i": int(cand_i["ID_CAND"]) if "ID_CAND" in candidates_df.columns else int(pos_i),
                "ID_CAND_j": int(cand_j["ID_CAND"]) if "ID_CAND" in candidates_df.columns else int(pos_j),
                "PAC_i": cand_i.get("PAC"),
                "PAC_j": cand_j.get("PAC"),
                "PENALIDADE_PAR": penalidade,
            }
        )

    objetivo_total = beneficio_total - float(alpha) * penalidade_total
    return {
        "objetivo_total": float(objetivo_total),
        "beneficio_total": beneficio_total,
        "penalidade_total": float(penalidade_total),
        "pares_penalizados": pares_penalizados,
    }


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
