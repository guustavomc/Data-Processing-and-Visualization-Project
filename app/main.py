import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
import warnings

import visualization
import data_quality
import encoding

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, "data", "NASA_Asteroid_Close_Approaches.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"]          = 110
plt.rcParams["axes.spines.top"]     = False
plt.rcParams["axes.spines.right"]   = False


# =============================================================================
# SEÇÃO 1 — DEFINIÇÃO DO PROBLEMA E CONTEXTUALIZAÇÃO
# =============================================================================
print("=" * 70)
print("SEÇÃO 1 — DEFINIÇÃO DO PROBLEMA E CONTEXTUALIZAÇÃO")
print("=" * 70)

dataframe_imported = pd.read_csv(DATA_FILE)
print(dataframe_imported.head(10))

dataframe = dataframe_imported.copy()
dataframe["cd_parsed"] = pd.to_datetime(dataframe["cd"], format="%Y-%b-%d %H:%M", errors="coerce")
dataframe["year"]   = dataframe["cd_parsed"].dt.year
dataframe["month"]  = dataframe["cd_parsed"].dt.month
dataframe["is_pha"] = dataframe["h"] < 22          # flag de Potentially Hazardous Asteroid
dataframe["dist_ld"] = dataframe["dist"] * 389.17  # distância em Distâncias Lunares

print(f"\nShape do dataset: {dataframe.shape[0]} linhas x {dataframe.shape[1]} colunas")
print("\nPrimeiras linhas:")
print(dataframe_imported.head())

print("""
DESCRIÇÃO DOS ATRIBUTOS:
  des       — Designação do objeto (nome/número do asteroide)        [categórico]
  orbit_id  — ID da solução orbital usada no cálculo                 [categórico]
  jd        — Data juliana da aproximação                            [numérico]
  cd        — Data e hora da aproximação (Close-Approach Date)       [datetime]
  dist      — Distância nominal de aproximação em AU                 [numérico]
  dist_min  — Distância mínima de aproximação em AU                  [numérico]
  dist_max  — Distância máxima de aproximação em AU                  [numérico]
  v_rel     — Velocidade relativa à Terra no ponto de aproximação    [numérico]
  v_inf     — Velocidade no infinito (velocidade hiperbólica)        [numérico]
  t_sigma_f — Incerteza (3-sigma) no tempo de aproximação            [categórico]
  h         — Magnitude absoluta H (quanto menor, maior o objeto)   [numérico]

VARIÁVEIS-ALVO CANDIDATAS PARA ML SUPERVISIONADO:
  • is_pha (derivada de h < 22): classificação binária de asteroides
    Potencialmente Perigosos — alvo natural para classificação supervisionada.
  • dist < 0.05 AU: flag de proximidade crítica, também binária.
  Ambas permitem treinar modelos como Random Forest ou SVM para prever
  o risco de um asteroide com base em dist, v_rel, v_inf e h.

PERGUNTA ANALÍTICA CENTRAL:
  "É possível distinguir asteroides Potencialmente Perigosos (PHAs) dos
  demais com base nas características de sua aproximação com a Terra
  — distância, velocidade e magnitude absoluta?"
""")

print("\nEstatísticas descritivas:")
print(dataframe[["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h"]].describe().round(3).to_string())

# =============================================================================
# SEÇÃO 2 — ANÁLISE EXPLORATÓRIA E VISUALIZAÇÃO
# =============================================================================
visualization.generate_graphs(dataframe, OUTPUT_DIR)

# =============================================================================
# SEÇÃO 3 — ANÁLISE DA QUALIDADE DOS DADOS
# =============================================================================
data_quality.data_validation(dataframe, dataframe_imported, OUTPUT_DIR)

# =============================================================================
# SEÇÃO 4 — CODIFICAÇÃO DE ATRIBUTOS CATEGÓRICOS
# =============================================================================
encoding.run(dataframe, OUTPUT_DIR)