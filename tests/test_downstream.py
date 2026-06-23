from __future__ import annotations

import networkx as nx
import pandas as pd

from recloser_opt.downstream import calcular_metricas_nos


def test_calcula_metricas_jusante_em_arvore_radial() -> None:
    T = nx.DiGraph()
    T.add_edge("R", "A", COMP=10.0, NUM_FASES=3, TIPO="CP")
    T.add_edge("A", "B", COMP=5.0, NUM_FASES=3, TIPO="CP")
    cargas = pd.DataFrame(
        [
            {"PAC": "A", "UCs_A": 1, "UCs_B": 2, "DIC": 3.0, "FIC": 4.0},
            {"PAC": "B", "UCs_A": 0, "UCs_B": 5, "DIC": 6.0, "FIC": 7.0},
        ]
    )

    df = calcular_metricas_nos(T, "R", cargas).set_index("PAC")

    assert df.loc["B", "UCs_JUS"] == 5
    assert df.loc["A", "UCs_JUS"] == 8
    assert df.loc["R", "UCs_JUS"] == 8
    assert df.loc["B", "DIST_RAIZ"] == 15.0


def test_calcula_metricas_em_rede_profunda_sem_recursao() -> None:
    T = nx.DiGraph()
    total_trechos = 1500

    for i in range(total_trechos):
        T.add_edge(f"N{i}", f"N{i + 1}", COMP=1.0, NUM_FASES=3, TIPO="CP")

    cargas = pd.DataFrame(
        [
            {
                "PAC": f"N{total_trechos}",
                "UCs_A": 0,
                "UCs_B": 1,
                "DIC": 2.0,
                "FIC": 3.0,
            }
        ]
    )

    df = calcular_metricas_nos(T, "N0", cargas).set_index("PAC")

    assert len(df) == total_trechos + 1
    assert df.loc["N0", "UCs_JUS"] == 1
    assert df.loc["N0", "DIC_JUS"] == 2.0
    assert df.loc[f"N{total_trechos}", "DIST_RAIZ"] == float(total_trechos)
