import pandas as pd
import numpy as np
import os

abs_path = os.path.abspath(__file__)

rede = pd.read_csv(os.path.join(os.path.dirname(abs_path), 'dados_entrada\\otimizar_056001.csv'), sep=';')
print(rede)
