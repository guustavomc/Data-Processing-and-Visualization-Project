import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_graphs(dataframe, output_dir):
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
    plt.savefig(os.path.join(output_dir, "2_1_distribuicao_distancias.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_2_distribuicao_velocidades.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_3_distribuicao_magnitude_h.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_4_aproximacoes_por_ano.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_5_scatter_dist_velocidade.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_6_barras_t_sigma_f.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "2_7_correlacao.png"), bbox_inches="tight")
    plt.show()
    print("  Interpretação: dist_min e dist_max têm correlação muito alta com dist "
        "(esperado). v_rel e v_inf são fortemente correlacionados entre si. "
        "h tem correlação negativa com distância — objetos maiores tendem a ser "
        "detectados em distâncias maiores.")