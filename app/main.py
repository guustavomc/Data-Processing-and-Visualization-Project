import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
import warnings

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
print("\n" + "=" * 70)
print("SEÇÃO 2 — ANÁLISE EXPLORATÓRIA E VISUALIZAÇÃO")
print("=" * 70)

# --- 2.1 Distribuição das distâncias ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("2.1 Distribuição das Distâncias de Aproximação", fontsize=13)

axes[0].hist(dataframe["dist"].dropna(), bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
axes[0].axvline(0.05, color="red", linestyle="--", linewidth=1.2, label="Limite 0,05 AU")
axes[0].set_xlabel("Distância (AU)")
axes[0].set_ylabel("Número de Aproximações")
axes[0].set_title("Distribuição em AU")
axes[0].legend()

axes[1].hist(dataframe["dist_ld"].dropna(), bins=60, color="darkorange", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Distância (Distâncias Lunares)")
axes[1].set_ylabel("Número de Aproximações")
axes[1].set_title("Distribuição em Distâncias Lunares (LD)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_1_distribuicao_distancias.png"), bbox_inches="tight")
plt.show()
print(f"  Aproximações < 1 LD : {(dataframe['dist_ld'] < 1).sum()}")
print(f"  Aproximações < 10 LD: {(dataframe['dist_ld'] < 10).sum()}")
print("  Interpretação: distribuição assimétrica à direita — eventos muito "
      "próximos (<1 LD) são raros e de alto interesse científico.")

# --- 2.2 Distribuição das velocidades ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("2.2 Distribuição das Velocidades", fontsize=13)

axes[0].hist(dataframe["v_rel"].dropna(), bins=60, color="mediumseagreen", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("v_rel (km/s)")
axes[0].set_ylabel("Número de Aproximações")
axes[0].set_title("Velocidade Relativa à Terra")

axes[1].hist(dataframe["v_inf"].dropna(), bins=60, color="mediumpurple", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("v_inf (km/s)")
axes[1].set_ylabel("Número de Aproximações")
axes[1].set_title("Velocidade no Infinito")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_2_distribuicao_velocidades.png"), bbox_inches="tight")
plt.show()
print("  Interpretação: ambas as velocidades têm cauda longa à direita. "
      "Objetos com v_rel > 40 km/s são eventos extremos.")

# --- 2.3 Distribuição da Magnitude Absoluta H ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(dataframe["h"].dropna(), bins=60, color="indianred", edgecolor="white", linewidth=0.4)
ax.axvline(22, color="black", linestyle="--", linewidth=1.3, label="H = 22 (limiar PHA)")
ax.set_xlabel("Magnitude Absoluta H")
ax.set_ylabel("Número de Asteroides")
ax.set_title("2.3 Distribuição da Magnitude Absoluta H")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_3_distribuicao_magnitude_h.png"), bbox_inches="tight")
plt.show()
pha_count = dataframe["is_pha"].sum()
print(f"  PHAs (h < 22): {pha_count} ({100*pha_count/len(dataframe):.1f}% do total)")
print("  Interpretação: quanto menor H, maior o objeto. A linha em H=22 "
      "separa asteroides potencialmente perigosos dos demais.")

# --- 2.4 Tendência temporal — aproximações por ano ---
year_counts = dataframe["year"].dropna().astype(int).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(year_counts.index, year_counts.values, color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Ano")
ax.set_ylabel("Número de Aproximações")
ax.set_title("2.4 Aproximações por Ano")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_4_aproximacoes_por_ano.png"), bbox_inches="tight")
plt.show()
print("  Interpretação: o crescimento de registros nos anos recentes reflete "
      "melhorias nos programas de detecção, não um aumento real de eventos.")

# --- 2.5 Scatter: Distância × Velocidade (colorido por PHA) ---
fig, ax = plt.subplots(figsize=(10, 5))
colors = dataframe["is_pha"].map({True: "crimson", False: "steelblue"})
ax.scatter(dataframe["dist"], dataframe["v_rel"], c=colors, alpha=0.3, s=8, linewidths=0)
ax.set_xlabel("Distância (AU)")
ax.set_ylabel("Velocidade Relativa (km/s)")
ax.set_title("2.5 Distância × Velocidade — PHAs destacados")
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",    markersize=7, label="PHA (H < 22)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",  markersize=7, label="Não-PHA"),
]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_5_scatter_dist_velocidade.png"), bbox_inches="tight")
plt.show()
print("  Interpretação: PHAs tendem a se concentrar em distâncias menores. "
      "Não há separação clara por velocidade, sugerindo que distância e H "
      "são melhores discriminadores.")

# --- 2.6 Gráfico de barras — atributo categórico 't_sigma_f' ---
sigma_counts = dataframe["t_sigma_f"].value_counts().head(15).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(sigma_counts.index, sigma_counts.values, color="teal", edgecolor="white")
ax.set_xlabel("Incerteza no Tempo (t_sigma_f)")
ax.set_ylabel("Número de Aproximações")
ax.set_title("2.6 Distribuição da Incerteza Temporal (t_sigma_f) — Top 15")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_6_barras_t_sigma_f.png"), bbox_inches="tight")
plt.show()
print(f"  Valores únicos em 't_sigma_f': {dataframe['t_sigma_f'].nunique()}")
print("  Interpretação: a maioria das aproximações tem incerteza temporal muito "
      "baixa (< 1 hora), refletindo a precisão dos dados modernos da NASA. "
      "Valores altos indicam objetos detectados com menos observações.")

# --- 2.7 Mapa de correlação ---
num_cols = ["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h"]
corr = dataframe[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("2.7 Matriz de Correlação — Atributos Numéricos")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_7_correlacao.png"), bbox_inches="tight")
plt.show()
print("  Interpretação: dist_min e dist_max têm correlação muito alta com dist "
      "(esperado). v_rel e v_inf são fortemente correlacionados entre si. "
      "h tem correlação negativa com distância — objetos maiores tendem a ser "
      "detectados em distâncias maiores.")

# =============================================================================
# SEÇÃO 3 — ANÁLISE DA QUALIDADE DOS DADOS
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 3 — ANÁLISE DA QUALIDADE DOS DADOS")
print("=" * 70)

# --- 3.1 Dados faltantes ---
missing       = dataframe_imported.isnull().sum()
missing_pct   = (missing / len(dataframe_imported) * 100).round(2)
missing_df    = pd.DataFrame({"Faltantes": missing, "% do Total": missing_pct})
missing_df    = missing_df[missing_df["Faltantes"] > 0].sort_values("% do Total", ascending=False)

print("\n3.1 Dados Faltantes:")
if missing_df.empty:
    print("  Nenhum valor faltante encontrado.")
else:
    print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
cols_all    = dataframe_imported.columns
miss_counts = dataframe_imported.isnull().sum()
colors_miss = ["crimson" if v > 0 else "steelblue" for v in miss_counts]
ax.bar(cols_all, miss_counts, color=colors_miss, edgecolor="white")
ax.set_xlabel("Coluna")
ax.set_ylabel("Valores Faltantes")
ax.set_title("3.1 Valores Faltantes por Coluna")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_1_dados_faltantes.png"), bbox_inches="tight")
plt.show()

# --- 3.2 Duplicatas ---
n_dup = dataframe_imported.duplicated().sum()
print(f"\n3.2 Linhas duplicadas: {n_dup} ({100*n_dup/len(dataframe_imported):.2f}% do total)")

# --- 3.3 Outliers via IQR ---
print("\n3.3 Análise de Outliers (método IQR):")
outlier_cols = ["dist", "v_rel", "v_inf", "h"]
outlier_summary = {}

for col in outlier_cols:
    series = dataframe[col].dropna()
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    lower  = Q1 - 1.5 * IQR
    upper  = Q3 + 1.5 * IQR
    n_out  = ((series < lower) | (series > upper)).sum()
    outlier_summary[col] = {"Q1": Q1, "Q3": Q3, "IQR": IQR,
                             "Limite Inf": lower, "Limite Sup": upper,
                             "Outliers": n_out, "% Outliers": round(100*n_out/len(series), 2)}

outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df[["Limite Inf", "Limite Sup", "Outliers", "% Outliers"]].round(3).to_string())

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(14, 4))
fig.suptitle("3.3 Boxplots — Identificação de Outliers", fontsize=13)
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(dataframe[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_title(col)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_3_boxplots_outliers.png"), bbox_inches="tight")
plt.show()
print("  Interpretação: v_rel e v_inf apresentam os maiores percentuais de "
      "outliers, indicando eventos de alta velocidade atípicos. Esses valores "
      "são fisicamente plausíveis e não devem ser removidos sem critério.")

# =============================================================================
# SEÇÃO 4 — CODIFICAÇÃO DE ATRIBUTOS CATEGÓRICOS
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 4 — CODIFICAÇÃO DE ATRIBUTOS CATEGÓRICOS")
print("=" * 70)

cat_cols = ["t_sigma_f", "des"]
print(f"\nAtributos categóricos identificados: {cat_cols}")
print(f"\nValores únicos em 't_sigma_f': {dataframe['t_sigma_f'].nunique()}")
print(f"Valores únicos em 'des'      : {dataframe['des'].nunique()}")
print(f"\nDistribuição de 't_sigma_f' (top 10):\n{dataframe['t_sigma_f'].value_counts().head(10).to_string()}")

df_encoded = dataframe[["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h",
                         "t_sigma_f", "is_pha"]].copy()

# --- 4.1 Label Encoding em 't_sigma_f' ---
le = LabelEncoder()
df_encoded["t_sigma_f_label"] = le.fit_transform(df_encoded["t_sigma_f"].astype(str))
print(f"\n4.1 Label Encoding de 't_sigma_f' — primeiros 10 mapeamentos:")
for cls, lbl in zip(le.classes_[:10], le.transform(le.classes_[:10])):
    print(f"  '{cls}' → {lbl}")

# --- 4.2 Label Encoding em 'des' (alta cardinalidade — apenas label) ---
le_des = LabelEncoder()
df_encoded["des_label"] = le_des.fit_transform(dataframe["des"].astype(str))
print(f"\n4.2 Label Encoding de 'des' — {le_des.classes_.shape[0]} categorias únicas codificadas.")

# --- 4.3 One-Hot Encoding em 't_sigma_f' (após agrupar valores raros) ---
top_sigma = dataframe["t_sigma_f"].value_counts().head(10).index
df_encoded["t_sigma_grouped"] = dataframe["t_sigma_f"].where(
    dataframe["t_sigma_f"].isin(top_sigma), other="outro"
)
df_ohe = pd.get_dummies(df_encoded, columns=["t_sigma_grouped"], prefix="sigma", drop_first=False)
ohe_cols = [c for c in df_ohe.columns if c.startswith("sigma_")]
print(f"\n4.3 One-Hot Encoding de 't_sigma_f' (top-10 + 'outro') — colunas criadas: {ohe_cols}")

print(f"\nShape antes da codificação : {dataframe[['dist','v_rel','h','t_sigma_f']].shape}")
print(f"Shape após a codificação   : {df_ohe.shape}")

print("\nPrimeiras linhas do dataset codificado:")
print(df_ohe[["dist", "v_rel", "h", "t_sigma_f_label", "des_label"] + ohe_cols].head(5).to_string())

# Visualização comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("4 — Atributo 't_sigma_f': Antes e Após Codificação", fontsize=13)

sigma_top = dataframe["t_sigma_f"].value_counts().head(10)
axes[0].bar(sigma_top.index, sigma_top.values, color="teal", edgecolor="white")
axes[0].set_title("Antes — Valores Originais (string, top 10)")
axes[0].set_xlabel("t_sigma_f")
axes[0].set_ylabel("Contagem")
axes[0].tick_params(axis="x", rotation=45)

ohe_sums = df_ohe[ohe_cols].sum()
axes[1].bar(ohe_cols, ohe_sums.values, color="darkorange", edgecolor="white")
axes[1].set_title("Depois — One-Hot Encoding (numérico)")
axes[1].set_xlabel("Coluna binária")
axes[1].set_ylabel("Contagem de 1s")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_codificacao_categoricos.png"), bbox_inches="tight")
plt.show()

print("\nDataset final pronto para uso em modelos de ML supervisionado.")
print(f"Colunas finais: {list(df_ohe.columns)}")