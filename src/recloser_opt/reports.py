from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.recloser_opt.io_bdgd import resolver_saida  # noqa: E402

def salvar_resultados(
    alimentador: str,
    arestas_conectadas: pd.DataFrame,
    nos_metricas: pd.DataFrame,
    candidatos: pd.DataFrame,
    solucao: pd.DataFrame,
    info: dict[str, object],
    output_dir: str | Path | None = None,
) -> Path:
    pasta_saida = resolver_saida(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arestas_conectadas.to_csv(pasta_saida / f"{alimentador}_arestas_conectadas.csv", sep=";", index=False)
    nos_metricas.to_csv(pasta_saida / f"{alimentador}_nos_metricas.csv", sep=";", index=False)
    candidatos.to_csv(pasta_saida / f"{alimentador}_candidatos.csv", sep=";", index=False)
    solucao.to_csv(pasta_saida / f"{alimentador}_solucao_religadores.csv", sep=";", index=False)

    historico = info.get("historico")
    if isinstance(historico, pd.DataFrame):
        historico.to_csv(pasta_saida / f"{alimentador}_historico_ga.csv", sep=";", index=False)

    pares_redundantes = info.get("pares_redundantes")
    if isinstance(pares_redundantes, pd.DataFrame):
        pares_redundantes.to_csv(pasta_saida / f"{alimentador}_pares_redundantes.csv", sep=";", index=False)

    return pasta_saida


def imprimir_resumo_conectividade(resumo: dict[str, object]) -> None:
    print("\nResumo da conectividade:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")


def imprimir_solucao(solucao: pd.DataFrame, info: dict[str, object]) -> None:
    colunas = [
        "PAC",
        "PAI",
        "DIST_RAIZ",
        "UCs_A_JUS",
        "UCs_B_JUS",
        "UCs_JUS",
        "DIC_JUS",
        "FIC_JUS",
        "BENEFICIO",
        "TRONCO_AUTO",
        "INICIO_RAMAL",
        "BIFURCACAO",
        "FIM_RAMAL",
    ]
    colunas_existentes = [col for col in colunas if col in solucao.columns]

    print("\nSolucao encontrada:")
    print(solucao[colunas_existentes])
    print("\nResumo da otimizacao:")
    print(f"  Objetivo: {float(info['objetivo']):.6f}")
    print(f"  Beneficio total: {float(info['beneficio_total']):.6f}")
    print(f"  Penalidade total: {float(info['penalidade_total']):.6f}")

