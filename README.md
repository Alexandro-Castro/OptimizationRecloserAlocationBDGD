# Recloser Opt

Projeto para montar redes radiais a partir de CSVs da BDGD/ANEEL, calcular metricas a jusante e otimizar a alocacao de religadores por metaheuristica.

## Estrutura inicial

- `src/recloser_opt/io_bdgd.py`: leitura dos CSVs da BDGD preservando identificadores como texto.
- `src/recloser_opt/graph_builder.py`: montagem do grafo, componente conectada ao `PAC_INI` e orientacao radial.
- `src/recloser_opt/downstream.py`: metricas locais e a jusante por no.
- `src/recloser_opt/candidates.py`: preparacao dos pontos candidatos.
- `src/recloser_opt/objective.py`: penalidade de redundancia topologica.
- `src/recloser_opt/metaheuristics.py`: algoritmo genetico e funcoes legadas de Pareto/annealing.
- `src/recloser_opt/reports.py`: gravacao dos CSVs de saida.
- `src/recloser_opt/cli.py`: orquestracao do pipeline.
- `scripts/run_feeder.py`: script simples para rodar um alimentador.

## Dados

Os CSVs originais devem permanecer em `dados_entrada/`. Os resultados sao gravados em `saida_otimizacao/`.

Arquivos esperados:

- `CTMT.csv`
- `SSDMT.csv`
- `UNSEMT.csv`
- `UNREMT.csv`
- `UNTRMT.csv`
- `UCMT.csv`
- `UCBT.csv`

## Execucao

Rodar o alimentador `056001`:

```powershell
python .\scripts\run_feeder.py 056001 --n-religadores 3
```

Rodar o alimentador `056011`:

```powershell
python .\scripts\run_feeder.py 056011 --n-religadores 3
```

Para uma execucao curta de validacao do fluxo, reduza a populacao e as geracoes:

```powershell
python .\scripts\run_feeder.py 056001 --n-religadores 3 --pop-size 20 --geracoes 10
```

## Saidas

O pipeline gera, por alimentador:

- `<alimentador>_arestas_conectadas.csv`
- `<alimentador>_nos_metricas.csv`
- `<alimentador>_candidatos.csv`
- `<alimentador>_solucao_religadores.csv`
- `<alimentador>_historico_ga.csv`
- `<alimentador>_pares_redundantes.csv`

