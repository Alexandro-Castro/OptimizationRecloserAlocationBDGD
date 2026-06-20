from __future__ import annotations

import pandas as pd
from pathlib import Path

from .io_bdgd import CSV_DECIMAL_SEPARATOR, CSV_SEPARATOR, resolver_saida, resolver_saida_solucoes


def salvar_csv_saida(df: pd.DataFrame, caminho: Path) -> None:
    df.to_csv(
        caminho,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL_SEPARATOR,
        index=False,
    )

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

    salvar_csv_saida(arestas_conectadas, pasta_saida / f"{alimentador}_arestas_conectadas.csv")
    salvar_csv_saida(nos_metricas, pasta_saida / f"{alimentador}_nos_metricas.csv")
    salvar_csv_saida(candidatos, pasta_saida / f"{alimentador}_candidatos.csv")
    salvar_csv_saida(solucao, pasta_saida / f"{alimentador}_solucao_religadores.csv")

    historico = info.get("historico")
    if isinstance(historico, pd.DataFrame):
        salvar_csv_saida(historico, pasta_saida / f"{alimentador}_historico_ga.csv")

    pares_redundantes = info.get("pares_redundantes")
    if isinstance(pares_redundantes, pd.DataFrame):
        salvar_csv_saida(pares_redundantes, pasta_saida / f"{alimentador}_pares_redundantes.csv")

    return pasta_saida


def salvar_solucao_ga(
    alimentador: str,
    solucao: pd.DataFrame,
    historico: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    pasta_saida = resolver_saida_solucoes(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    salvar_csv_saida(solucao, pasta_saida / f"{alimentador}_solucao_ga.csv")
    salvar_csv_saida(historico, pasta_saida / f"{alimentador}_historico_ga.csv")
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
