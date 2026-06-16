from __future__ import annotations

import pandas as pd

from recloser_opt.objective import evaluate_solution, matriz_penalidade_redundancia


def _candidatos_objetivo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID_CAND": 0,
                "PAC": "A",
                "TIN": 1,
                "TOUT": 4,
                "DIST_RAIZ": 100.0,
                "UCs_JUS": 100.0,
                "BENEFICIO": 1.00,
            },
            {
                "ID_CAND": 1,
                "PAC": "B",
                "TIN": 2,
                "TOUT": 3,
                "DIST_RAIZ": 110.0,
                "UCs_JUS": 90.0,
                "BENEFICIO": 0.90,
            },
            {
                "ID_CAND": 2,
                "PAC": "C",
                "TIN": 4,
                "TOUT": 5,
                "DIST_RAIZ": 500.0,
                "UCs_JUS": 50.0,
                "BENEFICIO": 0.55,
            },
        ]
    )


def test_penalidade_so_aparece_para_candidatos_em_serie() -> None:
    cand = pd.DataFrame(
        [
            {"PAC": "A", "TIN": 1, "TOUT": 4, "DIST_RAIZ": 10.0, "UCs_JUS": 10.0},
            {"PAC": "B", "TIN": 2, "TOUT": 3, "DIST_RAIZ": 20.0, "UCs_JUS": 5.0},
            {"PAC": "C", "TIN": 4, "TOUT": 5, "DIST_RAIZ": 20.0, "UCs_JUS": 5.0},
        ]
    )

    P, HARD = matriz_penalidade_redundancia(cand, d0=1000.0, min_dist_serie=1.0)

    assert P[0, 1] > 0
    assert P[0, 2] == 0
    assert HARD.shape == P.shape


def test_dois_candidatos_em_ramais_diferentes_nao_tem_penalidade() -> None:
    cand = _candidatos_objetivo()
    P, _ = matriz_penalidade_redundancia(cand, d0=1000.0)

    resultado = evaluate_solution([1, 2], cand, P, alpha=1.0)

    assert P[1, 2] == 0
    assert resultado["penalidade_total"] == 0
    assert resultado["pares_penalizados"] == []


def test_dois_candidatos_em_serie_tem_penalidade() -> None:
    cand = _candidatos_objetivo()
    P, _ = matriz_penalidade_redundancia(cand, d0=1000.0)

    resultado = evaluate_solution([0, 1], cand, P, alpha=1.0)

    assert P[0, 1] > 0
    assert resultado["penalidade_total"] > 0
    assert len(resultado["pares_penalizados"]) == 1
    assert resultado["pares_penalizados"][0]["PAC_i"] == "A"
    assert resultado["pares_penalizados"][0]["PAC_j"] == "B"


def test_solucao_redundante_pode_perder_para_solucao_distribuida() -> None:
    cand = _candidatos_objetivo()
    P, _ = matriz_penalidade_redundancia(cand, d0=1000.0)

    redundante = evaluate_solution([0, 1], cand, P, alpha=1.0)
    distribuida = evaluate_solution([0, 2], cand, P, alpha=1.0)

    assert redundante["beneficio_total"] > distribuida["beneficio_total"]
    assert redundante["objetivo_total"] < distribuida["objetivo_total"]
