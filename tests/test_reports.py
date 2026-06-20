from __future__ import annotations

import pandas as pd

from recloser_opt.reports import salvar_solucao_ga


def test_salva_solucao_ga_em_saida_otimizacao_solucoes(tmp_path) -> None:
    output_root = tmp_path / "saida_otimizacao"
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

    pasta = salvar_solucao_ga("0001", solucao, historico, output_dir=output_root)

    assert pasta == output_root / "solucoes"
    assert (pasta / "0001_solucao_ga.csv").exists()
    assert (pasta / "0001_historico_ga.csv").exists()


def test_csvs_gerados_usam_virgula_decimal(tmp_path) -> None:
    output_root = tmp_path / "saida_otimizacao"
    solucao = pd.DataFrame([{"PAC": "A", "BENEFICIO": 1.25}])
    historico = pd.DataFrame([{"geracao": 0, "melhor_objetivo": 2.5}])

    pasta = salvar_solucao_ga("0001", solucao, historico, output_dir=output_root)

    conteudo_solucao = (pasta / "0001_solucao_ga.csv").read_text(encoding="utf-8")
    conteudo_historico = (pasta / "0001_historico_ga.csv").read_text(encoding="utf-8")
    assert "1,25" in conteudo_solucao
    assert "2,5" in conteudo_historico
