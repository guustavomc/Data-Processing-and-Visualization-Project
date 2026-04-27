import os
import matplotlib.pyplot as plt
import pandas as pd
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
    plt.savefig(os.path.join(output_dir, "3_3_boxplots_outliers.png"), bbox_inches="tight")
    plt.show()
    print("  Interpretação: v_rel e v_inf apresentam os maiores percentuais de "
        "outliers, indicando eventos de alta velocidade atípicos. Esses valores "
        "são fisicamente plausíveis e não devem ser removidos sem critério.")