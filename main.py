import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict, deque

caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dados_entrada')
alimentadores = ['056001','056011'] 

def construir_grafo(nos):

    grafo = defaultdict(set)

    for _, row in nos.iterrows():

        p1 = str(row['PAC_1'])
        p2 = str(row['PAC_2'])

        grafo[p1].add(p2)
        grafo[p2].add(p1)

    return grafo

def distancia_topologica(grafo, origem, destino, limite=25):

    if origem == destino:
        return 0

    visitados = {origem}

    fila = deque([(origem, 0)])

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

def penalizacao_proximidade(df_sol, grafo):

    penalizacao = 0

    pacs = df_sol['PAC_1'].astype(str).tolist()

    for pac1, pac2 in combinations(pacs, 2):

        dist = distancia_topologica(
            grafo,
            pac1,
            pac2,
            limite=50
        )

        if dist <= 50:

            penalizacao += (51 - dist)

    return penalizacao

def fronteira_pareto(df, objetivos):
    valores = df[objetivos].to_numpy()

    n = len(valores)
    pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            # j domina i?
            if np.all(valores[j] >= valores[i]) and np.any(valores[j] > valores[i]):
                pareto[i] = False
                break

    return df[pareto]

def score_grupo(df_sol, grafo):

    score_individual = (
        0.55 * df_sol['UCs']
        + 0.10 * df_sol['DEC']
        + 0.10 * df_sol['FEC']
        + 0.25 * df_sol['DIST_RAIZ']
    )

    balanceamento = (
        df_sol[['UCs','DEC','FEC','DIST_RAIZ']]
        .min(axis=1)
    )

    score_base = (
        score_individual * balanceamento
    ).sum()

    penalizacao = penalizacao_proximidade(
        df_sol,
        grafo
    )

    return score_base - 0.5 * penalizacao

def simulated_annealing(df_pareto,
                        grafo,
                        n_chaves=5,
                        T0=100,
                        Tf=0.001,
                        alpha=0.95,
                        iter_por_temp=100):

    df_pareto = df_pareto.reset_index(drop=True)

    indices = np.arange(len(df_pareto))

    solucao_atual = random.sample(list(indices), n_chaves)

    score_atual = score_grupo(
        df_pareto.loc[solucao_atual],
        grafo
    )

    melhor_solucao = solucao_atual.copy()
    melhor_score = score_atual

    T = T0

    while T > Tf:

        for _ in range(iter_por_temp):

            vizinho = solucao_atual.copy()

            posicao = random.randint(0, n_chaves-1)

            disponiveis = list(
                set(indices) - set(vizinho)
            )

            novo_indice = random.choice(disponiveis)

            vizinho[posicao] = novo_indice

            score_vizinho = score_grupo(
                df_pareto.loc[vizinho],
                grafo
            )

            delta = score_vizinho - score_atual

            if delta > 0:
                aceita = True
            else:
                aceita = np.random.rand() < np.exp(delta/T)

            if aceita:

                solucao_atual = vizinho
                score_atual = score_vizinho

                if score_atual > melhor_score:

                    melhor_score = score_atual
                    melhor_solucao = vizinho.copy()

        T *= alpha

    return (
        df_pareto.loc[melhor_solucao],
        melhor_score
    )

for alimentador in alimentadores:
    df = pd.read_csv(fr'{caminho}\otimizar_{alimentador}.csv',sep=';')
    nos = pd.read_csv(fr'{caminho}\nos_{alimentador}.csv')
    grafo = construir_grafo(nos)

    df['UCs'] = df['UCs_A_JUS'] + df['UCs_B_JUS']
    df['DEC'] = (df['DIC_JUS'] / df['UCs']).fillna(0)
    df['FEC'] = (df['FIC_JUS'] / df['UCs']).fillna(0)
    df = df.drop(columns=['UCs_A_JUS', 'UCs_B_JUS','DIC_JUS','FIC_JUS'])
    df = df[df['TRONCO'] == 0]
    df = df[df['CHAVE'] == 0]

    colunas = ['UCs', 'DEC', 'FEC', 'DIST_RAIZ']

    for col in colunas:
        minimo = df[col].min()
        maximo = df[col].max()

        if maximo != minimo:
            df[col] = (df[col] - minimo) / (maximo - minimo)
        else:
            df[col] = 0.0

    df_pareto = fronteira_pareto(df, colunas)

    # print(f'Total de soluções: {len(df)}')
    # print(f'Soluções na fronteira de Pareto: {len(df_pareto)}')
    # print(df_pareto)

    # pacs = df_pareto['PAC_1'].astype(str).tolist()
    # filtro = "PAC_1 IN (" + ", ".join("'" + pac + "'" for pac in pacs) + ")"
    # print(filtro)
    
    melhores_pontos, score = simulated_annealing(
    df_pareto,
    grafo,
    n_chaves=5
    )

    print(melhores_pontos[['PAC_1','UCs','DEC','FEC','DIST_RAIZ']])
    pacs = melhores_pontos['PAC_1'].astype(str).tolist()
    filtro = "PAC_1 IN (" + ", ".join("'" + pac + "'" for pac in pacs) + ")"
    print(filtro)