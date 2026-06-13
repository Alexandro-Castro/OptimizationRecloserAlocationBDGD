# AGENTS.md

## Contexto do projeto

Este projeto otimiza a alocação de religadores em redes de distribuição radiais usando dados extraídos da BDGD/ANEEL.

Os arquivos CSV principais são:

- CTMT.csv: dados do alimentador, incluindo PAC_INI.
- SSDMT.csv: trechos de média tensão, com PAC_1, PAC_2, FAS_CON, COMP e CT_COD_OP.
- UNSEMT.csv: chaves de média tensão.
- UNREMT.csv: reguladores de tensão, se existirem.
- UNTRMT.csv: transformadores MT/BT.
- UCMT.csv: unidades consumidoras de média tensão.
- UCBT.csv: unidades consumidoras de baixa tensão.

O objetivo é montar a rede elétrica conectada a partir do PAC_INI, calcular métricas a jusante de cada nó e otimizar a posição de N religadores usando metaheurística.

## Regras de implementação

- Preserve códigos de PAC, CTMT, COD_ID e alimentador como string.
- Não converta identificadores para inteiro se houver risco de perder zeros à esquerda.
- Não sobrescreva os arquivos CSV originais.
- Salve resultados em saida_otimizacao/.
- Escreva funções pequenas, testáveis e com type hints quando possível.
- Use pandas e networkx.
- Evite lógica escondida em notebooks.
- Sempre que alterar uma função central, crie ou atualize testes.
- Não implemente custo de religador na função objetivo nesta etapa.
- A otimização deve escolher exatamente N religadores, não “até N”, salvo instrução contrária.
- não faça com interface de linha de comando para rodar os scripts

## Modelo elétrico inicial

- A rede deve ser tratada como radial.
- Se houver componentes desconectadas, manter apenas a componente conectada ao PAC_INI.
- Se houver ciclos cadastrais, reportar o problema e, nesta etapa inicial, usar uma árvore orientada a partir do PAC_INI.
- O religador candidato em um nó representa a instalação no trecho de entrada desse nó, isto é, PAI -> PAC.

## Função objetivo inicial

Maximizar benefício de confiabilidade com penalização por redundância topológica.

Benefício sugerido:

BENEFICIO = 0.45 * DIC_JUS_N + 0.25 * FIC_JUS_N + 0.20 * UCs_JUS_N + 0.10 * TRONCO

Penalizar pares de religadores em série no mesmo caminho radial quando houver grande sobreposição das cargas a jusante.

## Critérios de aceite

- O projeto deve rodar com um alimentador de exemplo.
- Deve gerar CSV com rede conectada.
- Deve gerar CSV com métricas por nó.
- Deve gerar CSV com candidatos.
- Deve gerar CSV com solução dos religadores.
- Deve haver testes para montagem do grafo, cálculo a jusante e função objetivo.