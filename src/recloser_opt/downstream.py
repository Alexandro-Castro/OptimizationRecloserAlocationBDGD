from __future__ import annotations

import networkx as nx
import pandas as pd

from .io_bdgd import normaliza_id


def calcular_euler_tree(T: nx.DiGraph, no_raiz: str) -> tuple[dict[str, int], dict[str, int], list[str]]:
    tin: dict[str, int] = {}
    tout: dict[str, int] = {}
    ordem: list[str] = []
    tempo = 0

    pilha: list[tuple[str, bool]] = [(no_raiz, False)]
    while pilha:
        u, saida = pilha.pop()

        if saida:
            tout[u] = tempo
            continue

        if u in tin:
            continue

        tin[u] = tempo
        tempo += 1
        ordem.append(u)

        pilha.append((u, True))
        for v in reversed(list(T.successors(u))):
            pilha.append((v, False))

    return tin, tout, ordem


def iterar_pos_ordem_arvore(T: nx.DiGraph, no_raiz: str) -> list[str]:
    ordem: list[str] = []
    visitados: set[str] = set()
    pilha: list[tuple[str, bool]] = [(no_raiz, False)]

    while pilha:
        u, expandido = pilha.pop()

        if expandido:
            ordem.append(u)
            continue

        if u in visitados:
            continue

        visitados.add(u)
        pilha.append((u, True))
        for v in reversed(list(T.successors(u))):
            pilha.append((v, False))

    return ordem


def calcular_metricas_nos(T: nx.DiGraph, no_raiz: str, cargas: pd.DataFrame) -> pd.DataFrame:
    cargas = cargas.copy()
    cargas["PAC"] = cargas["PAC"].map(normaliza_id)
    carga_por_no = cargas.set_index("PAC").to_dict(orient="index")
    dist = nx.single_source_dijkstra_path_length(T, no_raiz, weight="COMP")

    dados: dict[str, dict[str, object]] = {}
    for no in T.nodes:
        local = carga_por_no.get(no, {})
        dados[no] = {
            "PAC": no,
            "UCs_A": float(local.get("UCs_A", 0.0)),
            "UCs_B": float(local.get("UCs_B", 0.0)),
            "DIC": float(local.get("DIC", 0.0)),
            "FIC": float(local.get("FIC", 0.0)),
            "DIST_RAIZ": float(dist.get(no, 0.0)),
            "GRAU_FILHOS": int(T.out_degree(no)),
            "GRAU_PAIS": int(T.in_degree(no)),
        }

    for no in iterar_pos_ordem_arvore(T, no_raiz):
        ucs_a_jus = float(dados[no]["UCs_A"])
        ucs_b_jus = float(dados[no]["UCs_B"])
        dic_jus = float(dados[no]["DIC"])
        fic_jus = float(dados[no]["FIC"])

        for filho in T.successors(no):
            ucs_a_jus += float(dados[filho]["UCs_A_JUS"])
            ucs_b_jus += float(dados[filho]["UCs_B_JUS"])
            dic_jus += float(dados[filho]["DIC_JUS"])
            fic_jus += float(dados[filho]["FIC_JUS"])

        dados[no]["UCs_A_JUS"] = ucs_a_jus
        dados[no]["UCs_B_JUS"] = ucs_b_jus
        dados[no]["UCs_JUS"] = ucs_a_jus + ucs_b_jus
        dados[no]["DIC_JUS"] = dic_jus
        dados[no]["FIC_JUS"] = fic_jus

    tin, tout, _ = calcular_euler_tree(T, no_raiz)
    linhas = []

    for no, d in dados.items():
        pais = list(T.predecessors(no))
        pai = pais[0] if pais else None

        if pai is not None:
            ed = T.get_edge_data(pai, no, default={})
            comp_entrada = ed.get("COMP", 0.0)
            tipo_entrada = ed.get("TIPO", "")
            fases_entrada = ed.get("NUM_FASES", 0)
            grau_filhos_pai = int(T.out_degree(pai))
        else:
            comp_entrada = 0.0
            tipo_entrada = ""
            fases_entrada = 0
            grau_filhos_pai = 0

        d["PAI"] = pai
        d["COMP_ENTRADA"] = comp_entrada
        d["TIPO_ENTRADA"] = tipo_entrada
        d["FASES_ENTRADA"] = fases_entrada
        d["GRAU_FILHOS_PAI"] = grau_filhos_pai
        d["TIN"] = tin[no]
        d["TOUT"] = tout[no]
        d["CARGA_LOCAL"] = 1 if (float(d["UCs_A"]) + float(d["UCs_B"])) > 0 else 0
        d["BIFURCACAO"] = 1 if int(d["GRAU_FILHOS"]) >= 2 else 0
        d["INICIO_RAMAL"] = 1 if grau_filhos_pai >= 2 else 0
        d["FIM_RAMAL"] = 1 if int(d["GRAU_FILHOS"]) == 0 else 0

        linhas.append(d)

    return pd.DataFrame(linhas)


def marcar_tronco_automatico(T: nx.DiGraph, no_raiz: str, df_nos: pd.DataFrame) -> pd.DataFrame:
    df_nos = df_nos.copy()
    folhas = df_nos[df_nos["GRAU_FILHOS"] == 0].copy()

    if folhas.empty:
        df_nos["TRONCO_AUTO"] = 0
        return df_nos

    folha_principal = folhas.sort_values("UCs_JUS", ascending=False)["PAC"].iloc[0]
    caminho_tronco = nx.shortest_path(T, source=no_raiz, target=folha_principal)
    df_nos["TRONCO_AUTO"] = df_nos["PAC"].isin(caminho_tronco).astype(int)
    return df_nos
