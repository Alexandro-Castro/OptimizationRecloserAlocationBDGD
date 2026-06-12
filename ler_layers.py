import pandas as pd
import geopandas as gpd
import pyogrio

dir_bdgd = r"C:\Users\Gustavo\Downloads\Energisa_MT_405_2024-12-31_V11_20251003-1011.gdb"

def ler_layer_bdgd(gdb_path: str, layer: str, filtros: dict = None) -> pd.DataFrame:

    layers_disponiveis = [l[0] for l in pyogrio.list_layers(gdb_path) if l[0] is not None]
    layer_map = {l.upper(): l for l in layers_disponiveis}
    layer_real = layer_map.get(layer.upper())

    if layer_real is None:
        disponiveis = "\n  ".join(sorted(layers_disponiveis))
        raise ValueError(
            f"Layer '{layer}' não encontrada no GDB.\n"
            f"Layers disponíveis:\n  {disponiveis}"
        )

    gdf = gpd.read_file(gdb_path, layer=layer_real, engine="pyogrio")

    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df.columns = [c.upper() for c in df.columns]

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(
        lambda col: col.str.replace('"', "", regex=False).str.strip()
    )

    if filtros:
        for coluna, valor in filtros.items():
            coluna = coluna.upper()
            if coluna in df.columns:
                df = df[df[coluna].astype(str).str.strip() == str(valor).strip()]
            else:
                raise KeyError(f"Coluna '{coluna}' não encontrada na layer '{layer_real}'.")

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":

    # # CTMT
    # colunas_ctmt = ["COD_ID", "NOME", "PAC_INI", "TEN_NOM", "UNI_TR_AT"]
    # df_ctmt = ler_layer_bdgd(dir_bdgd, "CTMT")
    # colunas_ctmt_validas = [c for c in colunas_ctmt if c in df_ctmt.columns]
    # df_ctmt = df_ctmt[colunas_ctmt_validas]
    # df_ctmt.to_csv("CTMT.csv", index=False)
    # print(f"CTMT exportado: {len(df_ctmt)} registros | colunas: {colunas_ctmt_validas}")

    # SSDMT
    colunas_ssdmt = ["CT_COD_OP", "PAC_1", "PAC_2", "FAS_CON", "TIP_CND", "COMP"]
    df_ssdmt = ler_layer_bdgd(dir_bdgd, "SSDMT")
    colunas_ssdmt_validas = [c for c in colunas_ssdmt if c in df_ssdmt.columns]
    df_ssdmt = df_ssdmt[colunas_ssdmt_validas]
    df_ssdmt.to_csv("SSDMT.csv", index=False)
    print(f"SSDMT exportado: {len(df_ssdmt)} registros | colunas: {colunas_ssdmt_validas}")

    # # SEGCON
    # colunas_  = ["COD_ID", "R1", "X1"]
    # df_segcon = ler_layer_bdgd(dir_bdgd, "SEGCON")
    # colunas_segcon_validas = [c for c in colunas_segcon if c in df_segcon.columns]
    # df_segcon = df_segcon[colunas_segcon_validas]
    # df_segcon.to_csv("SEGCON.csv", index=False)
    # print(f"SEGCON exportado: {len(df_segcon)} registros | colunas: {colunas_segcon_validas}")

    rede = pd.read_csv(r'C:\Projetos\Otimização\SSDMT.csv')
    cabo = pd.read_csv(r'C:\Projetos\Otimização\SEGCON.csv')

    df_merge = pd.merge(rede,cabo,left_on='TIP_CND',right_on="COD_ID")
    df_merge = df_merge.drop(columns=['TIP_CND','COD_ID'])
    df_merge.to_csv("SSDMT.csv", index=False)

    # # UNSEMT
    # colunas_unsemt = ["COD_ID", "PAC_1", "PAC_2", "FAS_CON", "TIP_UNID", "P_N_OPE", "CAP_ELO", "TLCD", "CTMT"]
    # df_unsemt = ler_layer_bdgd(dir_bdgd, "UNSEMT")
    # colunas_unsemt_validas = [c for c in colunas_unsemt if c in df_unsemt.columns]
    # df_unsemt = df_unsemt[colunas_unsemt_validas]
    # df_unsemt.to_csv("UNSEMT.csv", index=False)
    # print(f"UNSEMT exportado: {len(df_unsemt)} registros | colunas: {colunas_unsemt_validas}")

#     TEN_NOM = {
#     0:   0,
#     1:   110,
#     2:   115,
#     3:   120,
#     4:   121,
#     5:   125,
#     6:   127,
#     7:   208,
#     8:   216,
#     9:   216.5,
#     10:  220,
#     11:  230,
#     12:  231,
#     13:  240,
#     14:  254,
#     15:  380,
#     16:  400,
#     17:  440,
#     18:  480,
#     19:  500,
#     20:  600,
#     21:  750,
#     22:  1000,
#     23:  2300,
#     24:  3200,
#     25:  3600,
#     26:  3785,
#     27:  3800,
#     28:  3848,
#     29:  3985,
#     30:  4160,
#     31:  4200,
#     32:  4207,
#     33:  4368,
#     34:  4560,
#     35:  5000,
#     36:  6000,
#     37:  6600,
#     38:  6930,
#     39:  7960,
#     40:  8670,
#     41:  11400,
#     42:  11900,
#     43:  12000,
#     44:  12600,
#     45:  12700,
#     46:  13200,
#     47:  13337,
#     48:  13530,
#     49:  13800,
#     50:  13860,
#     51:  14140,
#     52:  14190,
#     53:  14400,
#     54:  14835,
#     55:  15000,
#     56:  15200,
#     57:  19053,
#     58:  19919,
#     59:  21000,
#     60:  21500,
#     61:  22000,
#     62:  23000,
#     63:  23100,
#     64:  23827,
#     65:  24000,
#     66:  24200,
#     67:  25000,
#     68:  25800,
#     69:  27000,
#     70:  30000,
#     71:  33000,
#     72:  34500,
#     73:  36000,
#     74:  38000,
#     75:  40000,
#     76:  44000,
#     77:  45000,
#     78:  45400,
#     79:  48000,
#     80:  60000,
#     81:  66000,
#     82:  69000,
#     83:  72500,
#     84:  88000,
#     85:  88200,
#     86:  92000,
#     87:  100000,
#     88:  120000,
#     89:  121000,
#     90:  123000,
#     91:  131600,
#     92:  131630,
#     93:  131635,
#     94:  138000,
#     95:  145000,
#     96:  230000,
#     97:  345000,
#     98:  500000,
#     99:  750000,
#     100: 1000000,
#     101: 245000,
#     102: 550000,
#     103: 11000,
#     104: 11500,
#     105: 13000,
#     106: 20000,
#     107: 68000,
#     108: 85000,
#     109: 440000,
# }
    
#     ctmt = pd.read_csv(r'C:\Projetos\Otimização\CTMT.csv')
#     ctmt['TENSAO'] = ctmt['TEN_NOM'].map(TEN_NOM)/1000
#     ctmt.to_csv("CTMT.csv", index=False)
