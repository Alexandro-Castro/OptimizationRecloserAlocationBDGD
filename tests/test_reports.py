from __future__ import annotations

import pandas as pd

from recloser_opt.reports import salvar_solucao_ga


def test_salva_solucao_ga_em_outputs_solucoes(tmp_path) -> None:
    solucao = pd.DataFrame(
        [
            {
                "PAC": "A",
                "PAI": "R",
                "DIST_RAIZ": 10.0,
                "UCs_JUS": 100.0,
                "DIC_JUS": 10.0,
                "FIC_JUS": 5.0,
                "BENEFICIO": 1.0,
            }
        ]
    )
    historico = pd.DataFrame(
        [
            {
                "geracao": 0,
                "melhor_objetivo": 1.0,
                "media_objetivo": 0.8,
                "melhor_beneficio": 1.0,
                "melhor_penalidade": 0.0,
            }
        ]
    )

    pasta = salvar_solucao_ga("0001", solucao, historico, output_dir=tmp_path)

    assert pasta == tmp_path / "solucoes"
    assert (pasta / "0001_solucao_ga.csv").exists()
    assert (pasta / "0001_historico_ga.csv").exists()

