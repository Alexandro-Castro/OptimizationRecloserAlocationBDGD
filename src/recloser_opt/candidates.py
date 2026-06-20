from __future__ import annotations

import numpy as np
import pandas as pd

from .io_bdgd import norm01


def marcar_tronco_por_maior_ucs_jus(df_nos: pd.DataFrame, no_raiz: str) -> pd.DataFrame:
    df_nos = df_nos.copy()

    if "PAC" not in df_nos.columns or "UCs_JUS" not in df_nos.columns:
        df_nos["TRONCO_AUTO"] = 0
        return df_nos

    if "FIM_RAMAL" in df_nos.columns:
        folhas = df_nos[df_nos["FIM_RAMAL"] == 1].copy()
    elif "PAI" in df_nos.columns:
        pais = set(df_nos["PAI"].dropna().astype(str))
        folhas = df_nos[~df_nos["PAC"].astype(str).isin(pais)].copy()
    else:
        folhas = df_nos.copy()

    if folhas.empty:
        df_nos["TRONCO_AUTO"] = 0
        return df_nos

    folha_principal = folhas.sort_values("UCs_JUS", ascending=False)["PAC"].iloc[0]
    caminho = {str(folha_principal)}

    if "PAI" in df_nos.columns:
        pai_por_pac = df_nos.set_index("PAC")["PAI"].to_dict()
        no_atual = folha_principal

        while no_atual != no_raiz:
            pai = pai_por_pac.get(no_atual)
            if pd.isna(pai) or pai is None:
                break

            caminho.add(str(pai))
            no_atual = pai

    df_nos["TRONCO_AUTO"] = df_nos["PAC"].astype(str).isin(caminho).astype(int)
    return df_nos


def preparar_candidatos(
    df_nos: pd.DataFrame,
    no_raiz: str,
    min_ucs_jus: int = 1,
    min_dist_raiz: float = 1.0,
    compactar_candidatos: bool = True,
    pesos_beneficio: dict[str, float] | None = None,
) -> pd.DataFrame:
    cand = df_nos.copy()

    if "TRONCO_AUTO" not in cand.columns:
        cand = marcar_tronco_por_maior_ucs_jus(cand, no_raiz)

    cand = cand[cand["PAC"] != no_raiz].copy()
    cand = cand[cand["UCs_JUS"] >= min_ucs_jus].copy()
    cand = cand[cand["DIST_RAIZ"] >= min_dist_raiz].copy()

    if compactar_candidatos:
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
        0.45 * cand["DIC_JUS_N"]
        + 0.25 * cand["FIC_JUS_N"]
        + 0.20 * cand["UCs_JUS_N"]
        + 0.10 * cand["TRONCO_AUTO"]
    )

    cand = cand[cand["TRONCO_AUTO"] == 0].copy()
    cand = cand.sort_values("BENEFICIO", ascending=False).reset_index(drop=True)
    cand["ID_CAND"] = np.arange(len(cand))
    return cand
