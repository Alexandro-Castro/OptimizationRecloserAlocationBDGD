from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .io_bdgd import (
    ler_alm,
    ler_chaves,
    ler_linhas,
    ler_reguladores,
    normaliza_pacs,
    CSV_DECIMAL_SEPARATOR,
    CSV_SEPARATOR,
    resolver_saida_redes_conectadas,
    to_num,
)


def remover_arestas_invalidas(arestas: pd.DataFrame) -> pd.DataFrame:
    arestas = normaliza_pacs(arestas, ["PAC_1", "PAC_2"])
    arestas = arestas.dropna(subset=["PAC_1", "PAC_2"]).copy()

    pac_1 = arestas["PAC_1"].astype(str).str.strip()
    pac_2 = arestas["PAC_2"].astype(str).str.strip()
    mask_valida = (pac_1 != "") & (pac_2 != "") & (pac_1 != pac_2)

    return arestas[mask_valida].copy()


def montar_arestas_rede(
    alimentador: str,
    input_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    no_raiz, cod_al, ctmt = ler_alm(alimentador, input_dir=input_dir)

    linhas = ler_linhas(cod_al, input_dir=input_dir)
    chaves = ler_chaves(ctmt, input_dir=input_dir)
    reguladores = ler_reguladores(ctmt, input_dir=input_dir)

    arestas = pd.concat([linhas, chaves, reguladores], ignore_index=True)
    arestas = remover_arestas_invalidas(arestas)

    arestas["COMP"] = to_num(arestas["COMP"])
    arestas["NUM_FASES"] = pd.to_numeric(arestas["NUM_FASES"], errors="coerce").fillna(0).astype(int)
    arestas["TIPO"] = arestas["TIPO"].fillna("")

    a = arestas["PAC_1"].astype(str)
    b = arestas["PAC_2"].astype(str)
    arestas["_NO_A"] = np.where(a <= b, a, b)
    arestas["_NO_B"] = np.where(a <= b, b, a)

    arestas_g = (
        arestas.groupby(["_NO_A", "_NO_B"], as_index=False)
        .agg(
            PAC_1=("_NO_A", "first"),
            PAC_2=("_NO_B", "first"),
            COMP=("COMP", "max"),
            NUM_FASES=("NUM_FASES", "max"),
            TIPO=("TIPO", lambda x: "+".join(sorted(set(map(str, x))))),
        )
    )

    arestas_g = arestas_g[["PAC_1", "PAC_2", "NUM_FASES", "COMP", "TIPO"]].copy()
    return arestas_g, no_raiz, cod_al, ctmt


def gerar_grafo(arestas: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in arestas.itertuples(index=False):
        comp = float(row.COMP) if pd.notna(row.COMP) else 0.0
        G.add_edge(
            row.PAC_1,
            row.PAC_2,
            COMP=comp,
            NUM_FASES=int(row.NUM_FASES) if pd.notna(row.NUM_FASES) else 0,
            TIPO=row.TIPO,
        )
    return G


def filtrar_componente_da_raiz(
    arestas: pd.DataFrame,
    no_raiz: str,
) -> tuple[pd.DataFrame, nx.Graph, dict[str, int | bool]]:
    G = gerar_grafo(arestas)

    if no_raiz not in G.nodes:
        raise ValueError(
            f"O no raiz {no_raiz} nao aparece nas arestas da rede. "
            "Verifique CTMT.PAC_INI, SSDMT, UNSEMT e UNREMT."
        )

    componente = nx.node_connected_component(G, no_raiz)
    mask = arestas["PAC_1"].isin(componente) & arestas["PAC_2"].isin(componente)
    arestas_conn = arestas[mask].copy()
    G_conn = gerar_grafo(arestas_conn)

    resumo = {
        "nos_total": G.number_of_nodes(),
        "arestas_total": G.number_of_edges(),
        "nos_conectados_raiz": G_conn.number_of_nodes(),
        "arestas_conectadas_raiz": G_conn.number_of_edges(),
        "nos_descartados": G.number_of_nodes() - G_conn.number_of_nodes(),
        "arestas_descartadas": G.number_of_edges() - G_conn.number_of_edges(),
        "eh_arvore": nx.is_tree(G_conn),
        "ciclos_estimados": G_conn.number_of_edges() - G_conn.number_of_nodes() + 1,
    }
    resumo["qtd_ciclos_estimado"] = resumo["ciclos_estimados"]
    return arestas_conn, G_conn, resumo


def salvar_arestas_conectadas(
    alimentador: str,
    arestas_conectadas: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    pasta_saida = resolver_saida_redes_conectadas(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_saida = pasta_saida / f"{alimentador}_arestas_conectadas.csv"
    arestas_conectadas.to_csv(
        caminho_saida,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL_SEPARATOR,
        index=False,
    )
    return caminho_saida


def montar_rede_conectada(
    alimentador: str,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    salvar_csv: bool = True,
) -> dict[str, object]:
    arestas, no_raiz, cod_al, ctmt = montar_arestas_rede(alimentador, input_dir=input_dir)
    arestas_conn, G_conn, diagnostico = filtrar_componente_da_raiz(arestas, no_raiz)

    caminho_saida = None
    if salvar_csv:
        caminho_saida = salvar_arestas_conectadas(alimentador, arestas_conn, output_dir=output_dir)

    return {
        "alimentador": str(alimentador),
        "pac_ini": no_raiz,
        "nome": cod_al,
        "cod_id": ctmt,
        "arestas_conectadas": arestas_conn,
        "grafo": G_conn,
        "diagnostico": diagnostico,
        "caminho_saida": caminho_saida,
    }


def orientar_rede_radial(G: nx.Graph, no_raiz: str) -> nx.DiGraph:
    _, paths = nx.single_source_dijkstra(G, source=no_raiz, weight="COMP")

    T = nx.DiGraph()
    T.add_node(no_raiz)

    for no, path in paths.items():
        if no == no_raiz:
            continue

        for u, v in zip(path[:-1], path[1:]):
            if not T.has_edge(u, v):
                dados = G.get_edge_data(u, v, default={})
                T.add_edge(
                    u,
                    v,
                    COMP=float(dados.get("COMP", 0.0)),
                    NUM_FASES=int(dados.get("NUM_FASES", 0)),
                    TIPO=dados.get("TIPO", ""),
                )
    return T


def construir_grafo(nos: pd.DataFrame) -> dict[str, set[str]]:
    grafo: dict[str, set[str]] = defaultdict(set)
    for _, row in nos.iterrows():
        p1 = str(row["PAC_1"])
        p2 = str(row["PAC_2"])
        grafo[p1].add(p2)
        grafo[p2].add(p1)
    return grafo


def distancia_topologica(
    grafo: dict[str, set[str]],
    origem: str,
    destino: str,
    limite: int = 25,
) -> int:
    if origem == destino:
        return 0

    visitados = {origem}
    fila: deque[tuple[str, int]] = deque([(origem, 0)])

    while fila:
        no, dist = fila.popleft()
        if dist > limite:
            return limite + 1

        for vizinho in grafo[no]:
            if vizinho == destino:
                return dist + 1

            if vizinho not in visitados:
                visitados.add(vizinho)
                fila.append((vizinho, dist + 1))

    return limite + 1
