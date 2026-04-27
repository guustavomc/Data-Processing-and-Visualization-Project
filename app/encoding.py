import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run(dataframe, output_dir):
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
    plt.savefig(os.path.join(output_dir, "4_codificacao_categoricos.png"), bbox_inches="tight")
    plt.show()

    print("\nDataset final pronto para uso em modelos de ML supervisionado.")
    print(f"Colunas finais: {list(df_ohe.columns)}")
