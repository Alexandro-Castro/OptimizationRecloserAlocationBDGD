# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx

from pathlib import Path
from itertools import combinations
import de_para


CAMINHO = Path(__file__).resolve().parent


# ============================================================
# 1. Funções auxiliares
# ============================================================

def ler_csv(nome_arquivo: str) -> pd.DataFrame:
    """
    Lê CSV preservando identificadores como texto.
    Usa detecção automática de separador.
    Se seus arquivos forem sempre separados por ';', pode trocar por sep=';'.
    """
    arq = CAMINHO / nome_arquivo
    if not arq.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arq}")

    return pd.read_csv(
        arq,
        sep=None,
        engine="python",
        dtype=str
    )


def normaliza_id(valor):
    """
    Normaliza identificadores de nó/equipamento.
    Evita problemas como 056001 virar 56001 ou 12345.0 virar 12345.
    """
    if pd.isna(valor):
        return np.nan

    s = str(valor).strip()

    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    return s


def to_num(serie, default=0.0):
    """
    Converte série para número, tratando vírgula decimal.
    """
    return (
        serie.astype(str)
        .str.replace(",", ".", regex=False)
        .replace(["nan", "None", ""], np.nan)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(default)
    )


def normaliza_pacs(df: pd.DataFrame, cols=("PAC_1", "PAC_2")) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(normaliza_id)
    return df


def norm01(s: pd.Series) -> pd.Series:
    """
    Normaliza série entre 0 e 1.
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    maximo = s.max()
    if maximo <= 0:
        return pd.Series(0.0, index=s.index)
    return s / maximo


# ============================================================
# 2. Leitura dos dados da BDGD
# ============================================================

def ler_alm(alimentador: str):
    al = ler_csv("CTMT.csv")
    al["NOME"] = al["NOME"].map(normaliza_id)

    al = al[al["NOME"] == normaliza_id(alimentador)].copy()

    if al.empty:
        raise ValueError(f"Alimentador {alimentador} não encontrado em CTMT.csv")

    no_raiz = normaliza_id(al["PAC_INI"].iloc[0])
    cod_al = normaliza_id(al["NOME"].iloc[0])
    ctmt = normaliza_id(al["COD_ID"].iloc[0])

    return no_raiz, cod_al, ctmt


def ler_linhas(alimentador: str) -> pd.DataFrame:
    cod_mt = ler_csv("SSDMT.csv")
    cod_mt["CT_COD_OP"] = cod_mt["CT_COD_OP"].map(normaliza_id)

    cod_mt = cod_mt[cod_mt["CT_COD_OP"] == normaliza_id(alimentador)].copy()

    cod_mt = normaliza_pacs(cod_mt, ["PAC_1", "PAC_2"])
    cod_mt["NUM_FASES"] = cod_mt["FAS_CON"].map(de_para.fases_num)
    cod_mt["COMP"] = to_num(cod_mt["COMP"])

    cod_mt = cod_mt[["PAC_1", "PAC_2", "NUM_FASES", "COMP"]].copy()
    cod_mt["TIPO"] = "CP"

    return cod_mt


def ler_chaves(ctmt: str) -> pd.DataFrame:
    sec_mt = ler_csv("UNSEMT.csv")
    sec_mt["CTMT"] = sec_mt["CTMT"].map(normaliza_id)

    sec_mt = sec_mt[sec_mt["CTMT"] == normaliza_id(ctmt)].copy()

    # Mantém somente chaves fechadas.
    sec_mt = sec_mt[sec_mt["P_N_OPE"] == "F"].copy()

    sec_mt = normaliza_pacs(sec_mt, ["PAC_1", "PAC_2"])
    sec_mt["NUM_FASES"] = sec_mt["FAS_CON"].map(de_para.fases_num)
    sec_mt["TIPO"] = sec_mt["TIP_UNID"].map(de_para.tipo_chave)

    sec_mt = sec_mt[["PAC_1", "PAC_2", "NUM_FASES", "TIPO"]].copy()
    sec_mt["COMP"] = 0.0

    return sec_mt


def ler_reguladores(ctmt: str) -> pd.DataFrame:
    arq = CAMINHO / "UNREMT.csv"
    if not arq.exists():
        return pd.DataFrame(columns=["PAC_1", "PAC_2", "NUM_FASES", "TIPO", "COMP"])

    reg_mt = ler_csv("UNREMT.csv")
    reg_mt["CTMT"] = reg_mt["CTMT"].map(normaliza_id)

    reg_mt = reg_mt[reg_mt["CTMT"] == normaliza_id(ctmt)].copy()

    if reg_mt.empty:
        return pd.DataFrame(columns=["PAC_1", "PAC_2", "NUM_FASES", "TIPO", "COMP"])

    reg_mt = normaliza_pacs(reg_mt, ["PAC_1", "PAC_2"])
    reg_mt["NUM_FASES"] = reg_mt["FAS_CON"].map(de_para.fases_num)

    reg_mt = reg_mt[["PAC_1", "PAC_2", "NUM_FASES"]].copy()
    reg_mt["TIPO"] = "RT"
    reg_mt["COMP"] = 0.0

    return reg_mt


def ler_cargas_mt(ctmt: str) -> pd.DataFrame:
    ucmt = ler_csv("UCMT.csv")
    ucmt["CTMT"] = ucmt["CTMT"].map(normaliza_id)

    ucmt = ucmt[ucmt["CTMT"] == normaliza_id(ctmt)].copy()

    if ucmt.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    ucmt["PAC"] = ucmt["PAC"].map(normaliza_id)

    colunas_dic = [f"DIC_{i:02d}" for i in range(1, 13) if f"DIC_{i:02d}" in ucmt.columns]
    colunas_fic = [f"FIC_{i:02d}" for i in range(1, 13) if f"FIC_{i:02d}" in ucmt.columns]

    for col in colunas_dic + colunas_fic:
        ucmt[col] = to_num(ucmt[col])

    ucmt["DIC_MEDIO"] = ucmt[colunas_dic].mean(axis=1) if colunas_dic else 0.0
    ucmt["FIC_MEDIO"] = ucmt[colunas_fic].mean(axis=1) if colunas_fic else 0.0

    df_final = (
        ucmt.groupby("PAC", as_index=False)
        .agg(
            UCs=("COD_ID", "count"),
            DIC=("DIC_MEDIO", "sum"),
            FIC=("FIC_MEDIO", "sum"),
        )
    )

    df_final["GRUPO"] = "A"

    return df_final


def ler_cargas_bt(ctmt: str) -> pd.DataFrame:
    ucbt = ler_csv("UCBT.csv")
    ucbt["CTMT"] = ucbt["CTMT"].map(normaliza_id)

    ucbt = ucbt[ucbt["CTMT"] == normaliza_id(ctmt)].copy()

    if ucbt.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    colunas_dic = [f"DIC_{i:02d}" for i in range(1, 13) if f"DIC_{i:02d}" in ucbt.columns]
    colunas_fic = [f"FIC_{i:02d}" for i in range(1, 13) if f"FIC_{i:02d}" in ucbt.columns]

    for col in colunas_dic + colunas_fic:
        ucbt[col] = to_num(ucbt[col])

    ucbt["DIC_MEDIO"] = ucbt[colunas_dic].mean(axis=1) if colunas_dic else 0.0
    ucbt["FIC_MEDIO"] = ucbt[colunas_fic].mean(axis=1) if colunas_fic else 0.0

    ucbt["UNI_TR_MT"] = ucbt["UNI_TR_MT"].map(normaliza_id)

    ucbt_g = (
        ucbt.groupby("UNI_TR_MT", as_index=False)
        .agg(
            UCs=("COD_ID", "count"),
            DIC=("DIC_MEDIO", "sum"),
            FIC=("FIC_MEDIO", "sum"),
        )
    )

    trafos = ler_csv("UNTRMT.csv")
    trafos["CTMT"] = trafos["CTMT"].map(normaliza_id)
    trafos = trafos[trafos["CTMT"] == normaliza_id(ctmt)].copy()

    if trafos.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    trafos["COD_ID"] = trafos["COD_ID"].map(normaliza_id)
    trafos["PAC_1"] = trafos["PAC_1"].map(normaliza_id)

    trafos = trafos[["COD_ID", "PAC_1"]].copy()

    merge = pd.merge(
        trafos,
        ucbt_g,
        left_on="COD_ID",
        right_on="UNI_TR_MT",
        how="inner"
    )

    if merge.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    merge["PAC"] = merge["PAC_1"]

    df_final = (
        merge.groupby("PAC", as_index=False)
        .agg(
            UCs=("UCs", "sum"),
            DIC=("DIC", "sum"),
            FIC=("FIC", "sum"),
        )
    )

    df_final["GRUPO"] = "B"

    return df_final


def agrupa_cargas(ctmt: str) -> pd.DataFrame:
    ucmt = ler_cargas_mt(ctmt)
    ucbt = ler_cargas_bt(ctmt)

    df_concat = pd.concat([ucmt, ucbt], ignore_index=True)

    if df_concat.empty:
        return pd.DataFrame(columns=["PAC", "UCs_A", "UCs_B", "DIC", "FIC"])

    df_concat["PAC"] = df_concat["PAC"].map(normaliza_id)

    df_concat["UCs_A"] = np.where(df_concat["GRUPO"] == "A", df_concat["UCs"], 0)
    df_concat["UCs_B"] = np.where(df_concat["GRUPO"] == "B", df_concat["UCs"], 0)

    UCs = (
        df_concat.groupby("PAC", as_index=False)
        .agg(
            UCs_A=("UCs_A", "sum"),
            UCs_B=("UCs_B", "sum"),
            DIC=("DIC", "sum"),
            FIC=("FIC", "sum"),
        )
    )

    return UCs


# ============================================================
# 3. Montagem da rede conectada
# ============================================================

def montar_arestas_rede(alimentador: str) -> tuple[pd.DataFrame, str, str, str]:
    no_raiz, cod_al, ctmt = ler_alm(alimentador)

    linhas = ler_linhas(cod_al)
    chaves = ler_chaves(ctmt)
    reguladores = ler_reguladores(ctmt)

    arestas = pd.concat([linhas, chaves, reguladores], ignore_index=True)

    arestas = normaliza_pacs(arestas, ["PAC_1", "PAC_2"])

    arestas = arestas.dropna(subset=["PAC_1", "PAC_2"]).copy()
    arestas = arestas[arestas["PAC_1"] != arestas["PAC_2"]].copy()

    arestas["COMP"] = to_num(arestas["COMP"])
    arestas["NUM_FASES"] = pd.to_numeric(arestas["NUM_FASES"], errors="coerce").fillna(0).astype(int)
    arestas["TIPO"] = arestas["TIPO"].fillna("")

    # Cria chave não orientada para eliminar duplicidades entre os mesmos nós.
    a = arestas["PAC_1"].astype(str)
    b = arestas["PAC_2"].astype(str)

    arestas["_NO_A"] = np.where(a <= b, a, b)
    arestas["_NO_B"] = np.where(a <= b, b, a)

    # Para conectividade, uma ligação entre dois PACs deve aparecer uma vez.
    # COMP: usa maior comprimento representativo. Chaves/reguladores normalmente entram com COMP = 0.
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
            TIPO=row.TIPO
        )

    return G


def filtrar_componente_da_raiz(arestas: pd.DataFrame, no_raiz: str):
    """
    Mantém somente a componente conectada ao PAC_INI do alimentador.
    Isso elimina ilhas, sobras de cadastro e trechos desconectados.
    """
    G = gerar_grafo(arestas)

    if no_raiz not in G.nodes:
        raise ValueError(
            f"O nó raiz {no_raiz} não aparece nas arestas da rede. "
            f"Verifique CTMT.PAC_INI, SSDMT, UNSEMT e UNREMT."
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
        "qtd_ciclos_estimado": G_conn.number_of_edges() - G_conn.number_of_nodes() + 1,
    }

    return arestas_conn, G_conn, resumo


def orientar_rede_radial(G: nx.Graph, no_raiz: str) -> nx.DiGraph:
    """
    Orienta a rede da subestação para jusante.

    Se o grafo já for radial, isso preserva a topologia.
    Se houver ciclos por problema cadastral, usa árvore de menores caminhos a partir da raiz.
    """
    dist, paths = nx.single_source_dijkstra(G, source=no_raiz, weight="COMP")

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
                    TIPO=dados.get("TIPO", "")
                )

    return T


# ============================================================
# 4. Cálculo dos dados a jusante
# ============================================================

def calcular_euler_tree(T: nx.DiGraph, no_raiz: str):
    """
    Calcula tempos de entrada e saída para testar ancestralidade rapidamente.
    i é ancestral de j se TIN_i <= TIN_j < TOUT_i.
    """
    tin = {}
    tout = {}
    ordem = []
    tempo = 0

    def dfs(u):
        nonlocal tempo
        tin[u] = tempo
        tempo += 1
        ordem.append(u)

        for v in T.successors(u):
            dfs(v)

        tout[u] = tempo

    dfs(no_raiz)

    return tin, tout, ordem


def calcular_metricas_nos(T: nx.DiGraph, no_raiz: str, cargas: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula UCs, DIC e FIC locais e a jusante para cada nó.
    O candidato ao religador será o nó, representando instalar no trecho que chega nele.
    """
    cargas = cargas.copy()
    cargas["PAC"] = cargas["PAC"].map(normaliza_id)

    carga_por_no = cargas.set_index("PAC").to_dict(orient="index")

    # Distância da raiz pela árvore orientada.
    dist = nx.single_source_dijkstra_path_length(T, no_raiz, weight="COMP")

    # Inicializa métricas locais.
    dados = {}

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

    # Pós-ordem: soma dos filhos para o pai.
    for no in nx.dfs_postorder_nodes(T, source=no_raiz):
        ucs_a_jus = dados[no]["UCs_A"]
        ucs_b_jus = dados[no]["UCs_B"]
        dic_jus = dados[no]["DIC"]
        fic_jus = dados[no]["FIC"]

        for filho in T.successors(no):
            ucs_a_jus += dados[filho]["UCs_A_JUS"]
            ucs_b_jus += dados[filho]["UCs_B_JUS"]
            dic_jus += dados[filho]["DIC_JUS"]
            fic_jus += dados[filho]["FIC_JUS"]

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

        # Marcadores úteis para filtrar candidatos.
        d["CARGA_LOCAL"] = 1 if (d["UCs_A"] + d["UCs_B"]) > 0 else 0
        d["BIFURCACAO"] = 1 if d["GRAU_FILHOS"] >= 2 else 0
        d["INICIO_RAMAL"] = 1 if grau_filhos_pai >= 2 else 0
        d["FIM_RAMAL"] = 1 if d["GRAU_FILHOS"] == 0 else 0

        linhas.append(d)

    df_nos = pd.DataFrame(linhas)

    return df_nos


def marcar_tronco_automatico(T: nx.DiGraph, no_raiz: str, df_nos: pd.DataFrame) -> pd.DataFrame:
    """
    Marca um 'tronco automático' aproximado:
    caminho da raiz até a folha com maior UCs_JUS/DIC_JUS/FIC_JUS.

    Isso é uma aproximação. Se você tiver o campo real de tronco da BDGD ou da empresa,
    é melhor usar o campo real.
    """
    df_nos = df_nos.copy()

    folhas = df_nos[df_nos["GRAU_FILHOS"] == 0].copy()

    if folhas.empty:
        df_nos["TRONCO_AUTO"] = 0
        return df_nos

    folhas["PESO_FOLHA"] = (
        norm01(folhas["UCs_JUS"]) * 0.40
        + norm01(folhas["DIC_JUS"]) * 0.40
        + norm01(folhas["FIC_JUS"]) * 0.20
    )

    folha_principal = folhas.sort_values("PESO_FOLHA", ascending=False)["PAC"].iloc[0]

    caminho_tronco = nx.shortest_path(T, source=no_raiz, target=folha_principal)

    df_nos["TRONCO_AUTO"] = df_nos["PAC"].isin(caminho_tronco).astype(int)

    return df_nos


# ============================================================
# 5. Candidatos e função objetivo
# ============================================================

def preparar_candidatos(
    df_nos: pd.DataFrame,
    no_raiz: str,
    min_ucs_jus: int = 1,
    min_dist_raiz: float = 1.0,
    compactar_candidatos: bool = True,
    pesos_beneficio: dict | None = None,
) -> pd.DataFrame:
    """
    Prepara os pontos candidatos para religadores.

    O candidato é um PAC, interpretado como instalação no trecho que chega até esse PAC.
    """
    if pesos_beneficio is None:
        pesos_beneficio = {
            "DIC": 0.45,
            "FIC": 0.25,
            "UC": 0.20,
            "TRONCO": 0.10,
            "DIST": 0.00,
        }

    cand = df_nos.copy()

    # Não instala religador no nó raiz.
    cand = cand[cand["PAC"] != no_raiz].copy()

    # Candidato precisa proteger alguma carga a jusante.
    cand = cand[cand["UCs_JUS"] >= min_ucs_jus].copy()

    # Evita pontos colados na subestação ou com distância zero.
    cand = cand[cand["DIST_RAIZ"] >= min_dist_raiz].copy()

    if compactar_candidatos:
        # Remove pontos intermediários pouco informativos em cadeias longas.
        # Mantém início de ramal, bifurcação, ponto com carga local ou fim de ramal.
        mask_relevante = (
            (cand["INICIO_RAMAL"] == 1)
            | (cand["BIFURCACAO"] == 1)
            | (cand["CARGA_LOCAL"] == 1)
            | (cand["FIM_RAMAL"] == 1)
        )

        cand = cand[mask_relevante].copy()

    cand["UCs_JUS_N"] = norm01(cand["UCs_JUS"])
    cand["DIC_JUS_N"] = norm01(cand["DIC_JUS"])
    cand["FIC_JUS_N"] = norm01(cand["FIC_JUS"])
    cand["DIST_RAIZ_N"] = norm01(cand["DIST_RAIZ"])

    cand["BENEFICIO"] = (
        pesos_beneficio["DIC"] * cand["DIC_JUS_N"]
        + pesos_beneficio["FIC"] * cand["FIC_JUS_N"]
        + pesos_beneficio["UC"] * cand["UCs_JUS_N"]
        + pesos_beneficio["TRONCO"] * cand["TRONCO_AUTO"]
        + pesos_beneficio["DIST"] * cand["DIST_RAIZ_N"]
    )

    cand = cand[cand["BENEFICIO"] > 0].copy()

    cand = cand.sort_values("BENEFICIO", ascending=False).reset_index(drop=True)
    cand["ID_CAND"] = np.arange(len(cand))

    return cand


def eh_ancestral(tin_a, tout_a, tin_b):
    return tin_a <= tin_b < tout_a


def matriz_penalidade_redundancia(
    cand: pd.DataFrame,
    d0: float = 1000.0,
    peso_sobreposicao: float = 0.70,
    peso_proximidade: float = 0.30,
    min_dist_serie: float = 500.0,
    jaccard_hard: float = 0.80,
):
    """
    Calcula penalidade Pij entre candidatos.

    Em rede radial, dois candidatos só têm sobreposição de jusante quando um é ancestral do outro.
    Se estão em ramais diferentes, Pij = 0.
    """
    n = len(cand)

    P = np.zeros((n, n), dtype=float)
    HARD = np.zeros((n, n), dtype=bool)

    tin = cand["TIN"].to_numpy()
    tout = cand["TOUT"].to_numpy()
    dist = cand["DIST_RAIZ"].to_numpy()

    # Usa UCs_JUS como peso de região.
    # Se quiser, pode trocar por combinação de UCs/DIC/FIC.
    W = cand["UCs_JUS"].clip(lower=1.0).to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            i_anc_j = eh_ancestral(tin[i], tout[i], tin[j])
            j_anc_i = eh_ancestral(tin[j], tout[j], tin[i])

            if not i_anc_j and not j_anc_i:
                continue

            if i_anc_j:
                up, down = i, j
            else:
                up, down = j, i

            # Para conjuntos aninhados:
            # interseção = região do ponto mais a jusante.
            # união = região do ponto mais a montante.
            jaccard = W[down] / max(W[up], 1e-9)

            distancia_serie = abs(dist[i] - dist[j])
            proximidade = np.exp(-distancia_serie / max(d0, 1e-9))

            penalidade = (
                peso_sobreposicao * jaccard
                + peso_proximidade * proximidade
            )

            P[i, j] = penalidade
            P[j, i] = penalidade

            # Restrição dura: evita dois religadores muito próximos no mesmo caminho
            # protegendo praticamente a mesma região.
            if distancia_serie < min_dist_serie and jaccard >= jaccard_hard:
                HARD[i, j] = True
                HARD[j, i] = True

    return P, HARD


# ============================================================
# 6. Algoritmo genético
# ============================================================

def otimizar_religadores_ga(
    cand: pd.DataFrame,
    n_religadores: int,
    alpha_penalidade: float = 1.0,
    d0: float = 1000.0,
    min_dist_serie: float = 500.0,
    pop_size: int = 120,
    geracoes: int = 250,
    taxa_mutacao: float = 0.25,
    elite: int = 8,
    seed: int = 42,
):
    """
    Otimiza seleção de N religadores por GA.

    Como não há custo de religador, usa exatamente n_religadores.
    """
    if cand.empty:
        raise ValueError("Não há candidatos disponíveis para otimização.")

    if n_religadores <= 0:
        raise ValueError("n_religadores deve ser maior que zero.")

    if n_religadores > len(cand):
        raise ValueError(
            f"n_religadores={n_religadores} é maior que a quantidade de candidatos={len(cand)}."
        )

    rng = np.random.default_rng(seed)

    beneficio = cand["BENEFICIO"].to_numpy(dtype=float)

    P, HARD = matriz_penalidade_redundancia(
        cand,
        d0=d0,
        min_dist_serie=min_dist_serie,
    )

    hard_penalty = 1e6

    def criar_individuo():
        return np.sort(rng.choice(len(cand), size=n_religadores, replace=False))

    def reparar(ind):
        ind = list(dict.fromkeys(map(int, ind)))

        disponiveis = np.setdiff1d(np.arange(len(cand)), np.array(ind), assume_unique=False)

        while len(ind) < n_religadores:
            novo = int(rng.choice(disponiveis))
            ind.append(novo)
            disponiveis = np.setdiff1d(disponiveis, np.array([novo]), assume_unique=False)

        if len(ind) > n_religadores:
            ind = list(rng.choice(ind, size=n_religadores, replace=False))

        return np.sort(np.array(ind, dtype=int))

    def fitness(ind):
        ind = np.array(ind, dtype=int)

        valor = beneficio[ind].sum()

        penalidade = 0.0

        for a, b in combinations(ind, 2):
            penalidade += P[a, b]

            if HARD[a, b]:
                penalidade += hard_penalty

        return valor - alpha_penalidade * penalidade

    def torneio(pop, fits, k=3):
        idx = rng.choice(len(pop), size=k, replace=False)
        melhor = idx[np.argmax(fits[idx])]
        return pop[melhor]

    def crossover(p1, p2):
        pool = np.unique(np.concatenate([p1, p2]))

        if len(pool) >= n_religadores:
            filho = rng.choice(pool, size=n_religadores, replace=False)
        else:
            faltam = n_religadores - len(pool)
            disponiveis = np.setdiff1d(np.arange(len(cand)), pool, assume_unique=False)
            complemento = rng.choice(disponiveis, size=faltam, replace=False)
            filho = np.concatenate([pool, complemento])

        return reparar(filho)

    def mutar(ind):
        ind = ind.copy()

        if rng.random() < taxa_mutacao:
            pos = rng.integers(0, n_religadores)
            disponiveis = np.setdiff1d(np.arange(len(cand)), ind, assume_unique=False)

            if len(disponiveis) > 0:
                ind[pos] = int(rng.choice(disponiveis))

        return reparar(ind)

    # População inicial
    pop = [criar_individuo() for _ in range(pop_size)]

    melhor_ind = None
    melhor_fit = -np.inf

    historico = []

    for g in range(geracoes):
        fits = np.array([fitness(ind) for ind in pop])

        idx_melhor = int(np.argmax(fits))

        if fits[idx_melhor] > melhor_fit:
            melhor_fit = float(fits[idx_melhor])
            melhor_ind = pop[idx_melhor].copy()

        historico.append(
            {
                "geracao": g,
                "melhor_fitness": melhor_fit,
                "media_fitness": float(np.mean(fits)),
            }
        )

        # Elitismo
        elite_idx = np.argsort(fits)[-elite:]
        nova_pop = [pop[i].copy() for i in elite_idx]

        while len(nova_pop) < pop_size:
            p1 = torneio(pop, fits)
            p2 = torneio(pop, fits)

            filho = crossover(p1, p2)
            filho = mutar(filho)

            nova_pop.append(filho)

        pop = nova_pop

    solucao = cand.iloc[melhor_ind].copy().reset_index(drop=True)
    solucao["SELECIONADO"] = 1

    # Diagnóstico da solução
    penalidade_total = 0.0
    pares_redundantes = []

    for a, b in combinations(melhor_ind, 2):
        penalidade_total += P[a, b]

        if P[a, b] > 0:
            pares_redundantes.append(
                {
                    "PAC_i": cand.iloc[a]["PAC"],
                    "PAC_j": cand.iloc[b]["PAC"],
                    "PENALIDADE_PAR": P[a, b],
                    "HARD": bool(HARD[a, b]),
                    "DIST_i": cand.iloc[a]["DIST_RAIZ"],
                    "DIST_j": cand.iloc[b]["DIST_RAIZ"],
                    "UCs_JUS_i": cand.iloc[a]["UCs_JUS"],
                    "UCs_JUS_j": cand.iloc[b]["UCs_JUS"],
                }
            )

    info = {
        "objetivo": melhor_fit,
        "beneficio_total": float(solucao["BENEFICIO"].sum()),
        "penalidade_total": float(penalidade_total),
        "n_candidatos": len(cand),
        "n_religadores": n_religadores,
        "historico": pd.DataFrame(historico),
        "pares_redundantes": pd.DataFrame(pares_redundantes),
    }

    return solucao, info


# ============================================================
# 7. Pipeline completo
# ============================================================

def otimizar_alimentador(
    alimentador: str,
    n_religadores: int = 3,
    salvar_csv: bool = True,
    min_ucs_jus: int = 1,
    min_dist_raiz: float = 1.0,
    compactar_candidatos: bool = True,
    alpha_penalidade: float = 1.0,
    d0: float = 1000.0,
    min_dist_serie: float = 500.0,
):
    no_raiz, cod_al, ctmt = ler_alm(alimentador)

    print(f"\nProcessando alimentador {alimentador}")
    print(f"CTMT: {ctmt}")
    print(f"Nó raiz: {no_raiz}")

    arestas, no_raiz, cod_al, ctmt = montar_arestas_rede(alimentador)

    arestas_conn, G_conn, resumo = filtrar_componente_da_raiz(arestas, no_raiz)

    print("\nResumo da conectividade:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")

    if not resumo["eh_arvore"]:
        print(
            "\nAtenção: a rede conectada à raiz possui ciclos. "
            "Será usada uma árvore de menores caminhos a partir da subestação."
        )

    T = orientar_rede_radial(G_conn, no_raiz)

    cargas = agrupa_cargas(ctmt)

    df_nos = calcular_metricas_nos(T, no_raiz, cargas)
    df_nos = marcar_tronco_automatico(T, no_raiz, df_nos)

    candidatos = preparar_candidatos(
        df_nos,
        no_raiz=no_raiz,
        min_ucs_jus=min_ucs_jus,
        min_dist_raiz=min_dist_raiz,
        compactar_candidatos=compactar_candidatos,
    )

    print(f"\nQuantidade de nós conectados: {len(df_nos)}")
    print(f"Quantidade de candidatos: {len(candidatos)}")

    solucao, info = otimizar_religadores_ga(
        candidatos,
        n_religadores=n_religadores,
        alpha_penalidade=alpha_penalidade,
        d0=d0,
        min_dist_serie=min_dist_serie,
    )

    print("\nSolução encontrada:")
    print(
        solucao[
            [
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
        ]
    )

    print("\nResumo da otimização:")
    print(f"  Objetivo: {info['objetivo']:.6f}")
    print(f"  Benefício total: {info['beneficio_total']:.6f}")
    print(f"  Penalidade total: {info['penalidade_total']:.6f}")

    if salvar_csv:
        pasta_saida = CAMINHO / "saida_otimizacao"
        pasta_saida.mkdir(exist_ok=True)

        arestas_conn.to_csv(pasta_saida / f"{alimentador}_arestas_conectadas.csv", sep=";", index=False)
        df_nos.to_csv(pasta_saida / f"{alimentador}_nos_metricas.csv", sep=";", index=False)
        candidatos.to_csv(pasta_saida / f"{alimentador}_candidatos.csv", sep=";", index=False)
        solucao.to_csv(pasta_saida / f"{alimentador}_solucao_religadores.csv", sep=";", index=False)
        info["historico"].to_csv(pasta_saida / f"{alimentador}_historico_ga.csv", sep=";", index=False)
        info["pares_redundantes"].to_csv(pasta_saida / f"{alimentador}_pares_redundantes.csv", sep=";", index=False)

        print(f"\nArquivos salvos em: {pasta_saida}")

    return {
        "arestas_conectadas": arestas_conn,
        "grafo_conectado": G_conn,
        "arvore_orientada": T,
        "nos": df_nos,
        "candidatos": candidatos,
        "solucao": solucao,
        "info": info,
        "resumo_conectividade": resumo,
    }


# ============================================================
# 8. Execução
# ============================================================

if __name__ == "__main__":

    alimentadores = ["056001", "056011"]

    resultados = {}

    for alm in alimentadores:
        resultados[alm] = otimizar_alimentador(
            alimentador=alm,
            n_religadores=3,

            # Mínimo de UCs a jusante para ser candidato.
            min_ucs_jus=1,

            # Evita escolher ponto exatamente na raiz.
            min_dist_raiz=1.0,

            # Remove candidatos intermediários em cadeias pouco informativas.
            compactar_candidatos=True,

            # Quanto maior, mais o GA evita religadores em série/redundantes.
            alpha_penalidade=1.0,

            # Distância de decaimento da penalidade de proximidade.
            # Se a rede for rural, pode testar 2000, 3000, 5000 m.
            d0=1000.0,

            # Distância mínima para evitar religadores muito próximos no mesmo caminho.
            min_dist_serie=500.0,

            salvar_csv=True,
        )