import pandas as pd
import de_para
import numpy as np
from pathlib import Path
from collections import defaultdict

caminho = Path(__file__).resolve().parent

alimentadores = ['056001','056011'] 

def ler_alm(alimentador):
    al = pd.read_csv(fr'{caminho}\CTMT.csv')
    al = al[al["NOME"] == alimentador]
    
    no_raiz = al["PAC_INI"].values[0]
    cod_al = al["NOME"].values[0]
    ctmt = al["COD_ID"].values[0]
    
    return no_raiz, cod_al, ctmt

# print(al)
# print(ctmt)

def ler_linhas(alimentador):
    cod_mt = pd.read_csv(fr'{caminho}\SSDMT.csv')
    cod_mt = cod_mt[cod_mt["CT_COD_OP"] == alimentador]
    cod_mt['NUM_FASES'] = cod_mt['FAS_CON'].map(de_para.fases_num)
    cod_mt = cod_mt[['PAC_1','PAC_2','NUM_FASES','COMP']]
    cod_mt['TIPO'] = 'CP'
    return cod_mt

def ler_chaves(ctmt):
    sec_mt = pd.read_csv(fr'{caminho}\UNSEMT.csv')
    sec_mt = sec_mt[sec_mt['CTMT'] == ctmt]
    sec_mt = sec_mt[sec_mt['P_N_OPE'] == 'F']
    sec_mt['NUM_FASES'] = sec_mt['FAS_CON'].map(de_para.fases_num)
    sec_mt['TIPO'] = sec_mt['TIP_UNID'].map(de_para.tipo_chave)
    sec_mt = sec_mt[['PAC_1','PAC_2','NUM_FASES','TIPO']]
    sec_mt['COMP'] = 0
    return sec_mt

def ler_reguladores(ctmt):
    reg_mt = pd.read_csv(fr'{caminho}\UNREMT.csv')
    reg_mt = reg_mt[reg_mt["CTMT"] == ctmt]
    reg_mt['NUM_FASES'] = reg_mt['FAS_CON'].map(de_para.fases_num)
    reg_mt = reg_mt[['PAC_1','PAC_2','NUM_FASES']]
    reg_mt['TIPO'] = 'RT'
    reg_mt['COMP'] = 0
    return reg_mt

def ler_cargas_mt(ctmt):
    ucmt = pd.read_csv(fr'{caminho}\UCMT.csv')
    ucmt = ucmt[ucmt['CTMT'] == ctmt]
    
    colunas_dic = [f'DIC_{i:02d}' for i in range(1, 13)]
    colunas_fic = [f'FIC_{i:02d}' for i in range(1, 13)]
    
    ucmt['DIC_MEDIO'] = ucmt[colunas_dic].mean(axis=1)
    ucmt['FIC_MEDIO'] = ucmt[colunas_fic].mean(axis=1)
    
    df_final = ucmt.groupby(
        ['PAC']
    ).agg(
        UCs=('COD_ID', 'count'),
        DIC=('DIC_MEDIO','sum'),
        FIC=('FIC_MEDIO','sum'),
    ).reset_index()
    
    df_final['GRUPO'] = 'A'
    
    return df_final
    
def ler_cargas_bt(ctmt):
    ucbt = pd.read_csv(fr'{caminho}\UCBT.csv')
    ucbt = ucbt[ucbt['CTMT'] == ctmt]

    colunas_dic = [f'DIC_{i:02d}' for i in range(1, 13)]
    colunas_fic = [f'FIC_{i:02d}' for i in range(1, 13)]
    ucbt['DIC_MEDIO'] = ucbt[colunas_dic].mean(axis=1)
    ucbt['FIC_MEDIO'] = ucbt[colunas_fic].mean(axis=1)
    
    ucbt = ucbt.groupby(
        ['UNI_TR_MT']
    ).agg(
        UCs=('COD_ID', 'count'),
        DIC=('DIC_MEDIO','sum'),
        FIC=('FIC_MEDIO','sum'),
    ).reset_index()
    ucbt['UNI_TR_MT'] = ucbt['UNI_TR_MT'].astype(int)
    
    trafos = pd.read_csv(fr'{caminho}\UNTRMT.csv')
    trafos = trafos[trafos['CTMT'] == ctmt]
    trafos = trafos[['COD_ID','PAC_1']]
    trafos['COD_ID'] = trafos['COD_ID'].astype(int)
    
    merge = pd.merge(trafos,ucbt,left_on='COD_ID',right_on='UNI_TR_MT')
    merge = merge.drop(columns=['COD_ID','UNI_TR_MT'])
    merge['PAC'] = merge['PAC_1']
    
    df_final = merge.groupby(
        ['PAC']
    ).agg(
        UCs=('UCs', 'sum'),
        DIC=('DIC','sum'),
        FIC=('FIC','sum'),
    ).reset_index()
    
    df_final['GRUPO'] = 'B'
    
    return df_final

def agrupa_cargas(ctmt):
    ucmt = ler_cargas_mt(ctmt)
    ucbt = ler_cargas_bt(ctmt)
    df_concat = pd.concat([ucmt,ucbt], ignore_index=True)

    df_concat['UCs_A'] = df_concat['UCs'].where(df_concat['GRUPO'] == 'A', 0)
    df_concat['UCs_B'] = df_concat['UCs'].where(df_concat['GRUPO'] == 'B', 0)

    UCs = df_concat.groupby('PAC').agg(
        UCs_A=('UCs_A', 'sum'),
        UCs_B=('UCs_B', 'sum'),
        DIC=('DIC', 'sum'),
        FIC=('FIC', 'sum'),
    ).reset_index()
    
    return UCs

def agrupa_nos(alimentador,ctmt):
    trechos = ler_linhas(alimentador)
    chaves = ler_chaves(ctmt)
    regs = ler_reguladores(ctmt)
    
    nos = pd.concat([trechos,chaves,regs],ignore_index=True)
    nos.to_csv(fr'{caminho}\nos_{alimentador}.csv')
    
    return nos    
    
def classifica_tronco_ramal(nos, cargas, raiz):

    filhos = defaultdict(list)

    for idx, row in nos.iterrows():

        filhos[row['PAC_1']].append({
            'no': row['PAC_2'],
            'idx': idx,
            'comp': row['COMP']
        })

    cargas_idx = cargas.set_index('PAC')

    dict_ucs_a = cargas_idx['UCs_A'].to_dict()
    dict_ucs_b = cargas_idx['UCs_B'].to_dict()
    dict_dic   = cargas_idx['DIC'].to_dict()
    dict_fic   = cargas_idx['FIC'].to_dict()

    indicadores_jusante = {}

    def calcula_jusante(no):

        ucs_a = dict_ucs_a.get(no, 0)
        ucs_b = dict_ucs_b.get(no, 0)
        dic   = dict_dic.get(no, 0)
        fic   = dict_fic.get(no, 0)

        for filho in filhos.get(no, []):

            resultado = calcula_jusante(filho['no'])

            ucs_a += resultado['UCs_A']
            ucs_b += resultado['UCs_B']
            dic   += resultado['DIC']
            fic   += resultado['FIC']

        indicadores_jusante[no] = {
            'UCs_A': ucs_a,
            'UCs_B': ucs_b,
            'DIC': dic,
            'FIC': fic
        }

        return indicadores_jusante[no]

    calcula_jusante(raiz)

    distancia_raiz = {}

    def calcula_distancia_raiz(no, dist=0):

        distancia_raiz[no] = dist

        for filho in filhos.get(no, []):

            calcula_distancia_raiz(
                filho['no'],
                dist + filho['comp']
            )

    calcula_distancia_raiz(raiz)

    distancia_final = {}

    def calcula_distancia_final(no):

        filhos_no = filhos.get(no, [])

        if len(filhos_no) == 0:

            distancia_final[no] = distancia_raiz[no]

            return distancia_final[no]

        maior = 0

        for filho in filhos_no:

            maior = max(
                maior,
                calcula_distancia_final(filho['no'])
            )

        distancia_final[no] = maior

        return maior

    calcula_distancia_final(raiz)

    distancia_maxima_alimentador = max(
        distancia_raiz.values()
    )

    DIST_LIMITE = distancia_maxima_alimentador * 0.80
    MAX_DIF = distancia_maxima_alimentador * 0.05

    print(f'Distância máxima do alimentador: {distancia_maxima_alimentador:.0f} m')
    print(f'DIST_LIMITE: {DIST_LIMITE:.0f} m')
    print(f'MAX_DIF: {MAX_DIF:.0f} m')

    nos['TRONCO'] = 0

    def classifica(no, eh_tronco=True):

        filhos_no = filhos.get(no, [])

        if len(filhos_no) == 0:
            return

        if not eh_tronco:

            for filho in filhos_no:
                classifica(filho['no'], False)

            return

        maior_dist = max(
            distancia_final[filho['no']]
            for filho in filhos_no
        )

        dist_raiz_no = distancia_raiz.get(no, 0)

        filhos_tronco = []

        if dist_raiz_no <= DIST_LIMITE:

            for filho in filhos_no:

                if (
                    maior_dist
                    - distancia_final[filho['no']]
                ) <= MAX_DIF:

                    filhos_tronco.append(filho)

                    nos.loc[
                        filho['idx'],
                        'TRONCO'
                    ] = 1

        else:

            principal = max(
                filhos_no,
                key=lambda x:
                distancia_final[x['no']]
            )

            filhos_tronco.append(principal)

            nos.loc[
                principal['idx'],
                'TRONCO'
            ] = 1

        for filho in filhos_no:

            classifica(
                filho['no'],
                filho in filhos_tronco
            )

    classifica(raiz)

    nos['UCs_A_JUS'] = nos['PAC_2'].map(
        lambda x: indicadores_jusante.get(x, {}).get('UCs_A', 0)
    )

    nos['UCs_B_JUS'] = nos['PAC_2'].map(
        lambda x: indicadores_jusante.get(x, {}).get('UCs_B', 0)
    )

    nos['DIC_JUS'] = nos['PAC_2'].map(
        lambda x: indicadores_jusante.get(x, {}).get('DIC', 0)
    )

    nos['FIC_JUS'] = nos['PAC_2'].map(
        lambda x: indicadores_jusante.get(x, {}).get('FIC', 0)
    )

    nos['DIST_RAIZ'] = nos['PAC_2'].map(
        lambda x: distancia_raiz.get(x, 0)
    )

    # nos['DIST_FINAL'] = nos['PAC_2'].map(
    #     lambda x: distancia_final.get(x, 0)
    # )

    return nos, indicadores_jusante, distancia_raiz

for alimentador in alimentadores:
    raiz, al, ctmt = ler_alm(alimentador)
    nos = agrupa_nos(al, ctmt)
    cargas = agrupa_cargas(ctmt)

    nos, indicadores_jusante, distancia_raiz = classifica_tronco_ramal(
        nos,
        cargas,
        raiz
    )


    nos.to_csv(fr'{caminho}\otimizar_{alimentador}.csv',sep=';',index=False)

    # print('RAMAIS:')
    # print(', '.join(f"'{x}'" for x in nos_ot.loc[nos_ot['TRONCO'] == 0, 'PAC_1']))

    # print('\nTRONCO:')
    # print(', '.join(f"'{x}'" for x in nos_ot.loc[nos_ot['TRONCO'] == 1, 'PAC_1']))

