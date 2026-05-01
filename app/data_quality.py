import os
import matplotlib.pyplot as plt
import pandas as pd

def search_outliers(series: pd.Series) -> pd.Series:
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]

def data_validation(dataframe, dataframe_imported, output_dir):
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
    plt.savefig(os.path.join(output_dir, "3_1_dados_faltantes.png"), bbox_inches="tight")
    plt.show()

    df_treated = dataframe.copy()
    for col in ["v_inf", "h"]:
        if df_treated[col].isnull().sum() > 0:
            median = df_treated[col].median()
            df_treated[col] = df_treated[col].fillna(median)
            print(f"  '{col}': preenchidos com mediana ({median:.3f})")
    print(f"  Shape após tratamento: {df_treated.shape}")

    # --- 3.2 Duplicatas ---
    n_dup = dataframe_imported.duplicated().sum()
    print(f"\n3.2 Linhas duplicadas: {n_dup} ({100*n_dup/len(dataframe_imported):.2f}% do total)")

    df_sem_dup = dataframe_imported.drop_duplicates()
    print(f"  Shape antes: {dataframe_imported.shape} → após remoção: {df_sem_dup.shape}")

    # --- 3.3 Outliers via IQR ---
    print("\n3.3 Análise de Outliers (método IQR):")
    outlier_cols = ["dist", "v_rel", "v_inf", "h"]
    outlier_summary = {}
        
    for col in outlier_cols:
        series   = dataframe[col].dropna()
        outliers = search_outliers(series)
        Q1, Q3   = series.quantile(0.25), series.quantile(0.75)
        IQR      = Q3 - Q1
        outlier_summary[col] = {
            "Limite Inf": round(Q1 - 1.5 * IQR, 4),
            "Limite Sup": round(Q3 + 1.5 * IQR, 4),
            "Outliers":   len(outliers),
            "% Outliers": round(100 * len(outliers) / len(series), 2),
        }

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
    plt.savefig(os.path.join(output_dir, "3_3_boxplots_outliers.png"), bbox_inches="tight")
    plt.show()
    outliers_vrel   = search_outliers(dataframe["v_rel"].dropna())
    df_sem_outliers = dataframe.drop(index=outliers_vrel.index)
    print(f"\n  Exemplo de remoção em 'v_rel': {len(dataframe)} → {len(df_sem_outliers)} registros "
        f"({len(outliers_vrel)} outliers removidos)")
    print("  Nota: no contexto astronômico, outliers de velocidade são fisicamente "
        "plausíveis e não necessariamente devem ser removidos.")