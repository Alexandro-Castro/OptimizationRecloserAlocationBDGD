from __future__ import annotations

import pandas as pd

from recloser_opt.metaheuristics import genetic_algorithm_reclosers


def _candidatos_ga() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID_CAND": 10,
                "PAC": "A",
                "PAI": "R",
                "TIN": 1,
                "TOUT": 6,
                "DIST_RAIZ": 10.0,
                "UCs_JUS": 100.0,
                "DIC_JUS": 10.0,
                "FIC_JUS": 5.0,
                "DIC_JUS_N": 1.0,
                "FIC_JUS_N": 1.0,
                "UCs_JUS_N": 1.0,
                "TRONCO_AUTO": 1,
                "BENEFICIO": 1.0,
            },
            {
                "ID_CAND": 11,
                "PAC": "B",
                "PAI": "A",
                "TIN": 2,
                "TOUT": 3,
                "DIST_RAIZ": 20.0,
                "UCs_JUS": 80.0,
                "DIC_JUS": 8.0,
                "FIC_JUS": 4.0,
                "DIC_JUS_N": 0.8,
                "FIC_JUS_N": 0.8,
                "UCs_JUS_N": 0.8,
                "TRONCO_AUTO": 1,
                "BENEFICIO": 0.82,
            },
            {
                "ID_CAND": 12,
                "PAC": "C",
                "PAI": "A",
                "TIN": 3,
                "TOUT": 4,
                "DIST_RAIZ": 30.0,
                "UCs_JUS": 50.0,
                "DIC_JUS": 5.0,
                "FIC_JUS": 3.0,
                "DIC_JUS_N": 0.5,
                "FIC_JUS_N": 0.6,
                "UCs_JUS_N": 0.5,
                "TRONCO_AUTO": 0,
                "BENEFICIO": 0.475,
            },
            {
                "ID_CAND": 13,
                "PAC": "D",
                "PAI": "R",
                "TIN": 6,
                "TOUT": 8,
                "DIST_RAIZ": 15.0,
                "UCs_JUS": 70.0,
                "DIC_JUS": 7.0,
                "FIC_JUS": 4.0,
                "DIC_JUS_N": 0.7,
                "FIC_JUS_N": 0.8,
                "UCs_JUS_N": 0.7,
                "TRONCO_AUTO": 0,
                "BENEFICIO": 0.655,
            },
            {
                "ID_CAND": 14,
                "PAC": "E",
                "PAI": "D",
                "TIN": 7,
                "TOUT": 8,
                "DIST_RAIZ": 25.0,
                "UCs_JUS": 30.0,
                "DIC_JUS": 3.0,
                "FIC_JUS": 2.0,
                "DIC_JUS_N": 0.3,
                "FIC_JUS_N": 0.4,
                "UCs_JUS_N": 0.3,
                "TRONCO_AUTO": 0,
                "BENEFICIO": 0.295,
            },
        ]
    )


def test_ga_retorna_exatamente_n_candidatos() -> None:
    resultado = genetic_algorithm_reclosers(
        _candidatos_ga(),
        n_religadores=3,
        population_size=12,
        generations=8,
        mutation_rate=0.3,
        elite_size=2,
        random_seed=123,
    )

    assert len(resultado["selected_candidate_ids"]) == 3
    assert len(resultado["solution_df"]) == 3


def test_ga_nao_retorna_candidatos_repetidos() -> None:
    resultado = genetic_algorithm_reclosers(
        _candidatos_ga(),
        n_religadores=3,
        population_size=12,
        generations=8,
        mutation_rate=0.3,
        elite_size=2,
        random_seed=123,
    )

    selecionados = resultado["selected_candidate_ids"]
    assert len(selecionados) == len(set(selecionados))


def test_ga_com_seed_fixo_e_reprodutivel() -> None:
    kwargs = {
        "n_religadores": 3,
        "population_size": 12,
        "generations": 8,
        "mutation_rate": 0.3,
        "elite_size": 2,
        "random_seed": 123,
    }

    resultado_1 = genetic_algorithm_reclosers(_candidatos_ga(), **kwargs)
    resultado_2 = genetic_algorithm_reclosers(_candidatos_ga(), **kwargs)

    assert resultado_1["selected_candidate_ids"] == resultado_2["selected_candidate_ids"]
    assert resultado_1["objetivo_total"] == resultado_2["objetivo_total"]

