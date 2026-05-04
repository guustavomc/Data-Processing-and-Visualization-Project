import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def run(dataframe, output_dir):
    print("\n" + "=" * 70)
    print("SEÇÃO 4 — CODIFICAÇÃO DE ATRIBUTOS CATEGÓRICOS")
    print("=" * 70)

    cat_cols = ["t_sigma_f", "des"]
    print(f"\nAtributos categóricos identificados: {cat_cols}")
    print(f"\nValores únicos em 't_sigma_f': {dataframe['t_sigma_f'].nunique()}")
    print(f"Valores únicos em 'des'      : {dataframe['des'].nunique()}")
    print(f"\nDistribuição de 't_sigma_f' (top 10):\n{dataframe['t_sigma_f'].value_counts().head(10).to_string()}")

    print("\n  NOTA: Label Encoding de 'des' é demonstrativo. Em um modelo real,")
    print("  esta coluna seria descartada ou usada apenas como identificador,")
    print("  pois a ordem numérica imposta não tem significado semântico.")

    df_encoded = dataframe[["dist", "dist_min", "dist_max", "v_rel", "v_inf", "h",
                            "t_sigma_f", "des", "is_pha"]].copy()

    # --- 4.1 e 4.2 Label Encoding ---
    label_encoder = LabelEncoder()
    label_cols = ["t_sigma_f", "des"]
    for col in label_cols:
        df_encoded[f"{col}_encoded"] = label_encoder.fit_transform(df_encoded[col].astype(str))
        print(f"\n4.{label_cols.index(col)+1} Label Encoding de '{col}' — {label_encoder.classes_.shape[0]} categorias codificadas.")

    print(f"\nPrimeiros mapeamentos de 't_sigma_f':")
    label_encoder.fit(df_encoded["t_sigma_f"].astype(str))
    for cls, lbl in zip(label_encoder.classes_[:10], label_encoder.transform(label_encoder.classes_[:10])):
        print(f"  '{cls}' → {lbl}")

    # --- 4.3 One-Hot Encoding em 't_sigma_f' (após agrupar valores raros) ---
    top_sigma = dataframe["t_sigma_f"].value_counts().head(10).index
    df_encoded["t_sigma_grouped"] = dataframe["t_sigma_f"].where(
        dataframe["t_sigma_f"].isin(top_sigma), other="outro"
    )

    ohe = OneHotEncoder(sparse_output=False)
    sigma_encoded = ohe.fit_transform(df_encoded[["t_sigma_grouped"]])
    ohe_cols = ohe.get_feature_names_out(["t_sigma_grouped"]).tolist()
    sigma_encoded_df = pd.DataFrame(sigma_encoded, columns=ohe_cols, index=df_encoded.index)

    df_ohe = df_encoded.drop(columns=["t_sigma_grouped"])
    df_ohe = pd.concat([df_ohe, sigma_encoded_df], axis=1)
    print(f"\n4.3 One-Hot Encoding de 't_sigma_f' (top-10 + 'outro') — colunas criadas: {ohe_cols}")

    print(f"\nShape antes da codificação : {dataframe[['dist','v_rel','h','t_sigma_f']].shape}")
    print(f"Shape após a codificação   : {df_ohe.shape}")

    print("\nPrimeiras linhas do dataset codificado:")
    print(df_ohe[["dist", "v_rel", "h", "t_sigma_f_encoded", "des_encoded"] + ohe_cols].head(5).to_string())

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
