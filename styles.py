"""
styles.py — configuração central de estilo dos plots do PortFlow Invest.

Ponto de ajuste de paletas de cor, espessuras de linha e cores fixas
usadas pelos plots de `correlations.py` e `clustering.py`. Edite as constantes
aqui para testar paletas/espessuras sem tocar na lógica dos gráficos.
"""

import seaborn as sns

# Cor de fundo padrão das figuras
BACKGROUND = "#f8f9fa"


# Alternativa com cores distintas (descomente para testar):
MACRO_COLORS = {
    "macro_selic_change": "#8b00c4",  # Selic  — azul-petróleo
    "macro_ipca": "#e00029",  # IPCA   — terracota
    "macro_dollar_var": "#00ad07",  # Dólar  — verde
    "macro_ibcbr_var": "#2400d6",  # IBC-Br — roxo
}

MACRO_COLOR_FALLBACK = "#555555"


def macro_color(feature: str) -> str:
    """Cor fixa de um IMec; fallback cinza para chaves desconhecidas."""
    return MACRO_COLORS.get(feature, MACRO_COLOR_FALLBACK)


# =============================================================================
# Espessuras de linha
# =============================================================================

LINEWIDTHS = {
    "primary": 0.8,
    "secondary": 0.75,
    "emphasis": 1.1,
    "barycenter": 1.4,
    "member": 0.6,
    "evolution": 1.0,
    "reference": 1,
}


def lw(name: str) -> float:
    """Espessura de linha nomeada."""
    return LINEWIDTHS[name]


# =============================================================================
# Paletas categóricas por família de plot
#
# Cada valor é um nome de paleta seaborn/matplotlib (str) OU uma lista explícita
# de cores (hex).
# =============================================================================

PALETTES = {
    "clusters": "nipy_spectral",  # ids de cluster (DTW, overview, 3d, pairs, silhouette)
    "assets": "nipy_spectral",  # conjuntos de ativos (linhas asset-asset, preço bruto)
    "windows": "nipy_spectral",  # janelas (mesmo IMec em várias janelas)
    "lags": "nipy_spectral",  # defasagens (plot_feature_lag_comparison)
    "distribution": "nipy_spectral",  # boxplot / histograma da signature
}


def palette(name: str, n: int) -> list:
    """
    Resolve a paleta nomeada em uma lista de `n` cores. Aceita tanto um nome de
    paleta seaborn quanto uma lista explícita de cores (que é reciclada se `n`
    exceder seu tamanho).
    """
    spec = PALETTES[name]
    if isinstance(spec, (list, tuple)):
        return [spec[i % len(spec)] for i in range(n)]
    return sns.color_palette(spec, n)


# =============================================================================
# Cores semânticas avulsas (singletons)
# =============================================================================

COLORS = {
    "asset_emphasis": "#2c3e50",  # ativo-alvo nas séries brutas
    "neutral": "tab:gray",  # série base / Lag 0
    "member": "0.6",  # séries de fundo (cinza claro)
    "marker_edge": "black",  # borda dos marcadores de scatter
    "zero_line": "black",  # axhline em y=0
    "ref_line": "gray",  # axvline na ref_date
    "stripplot": "#34495e",  # pontos individuais sobre boxplot
    "sil_mean": "red",  # linha do silhouette médio
}

# Cores dos dois eixos gêmeos do sweep de k.
KSWEEP = {
    "silhouette": "#1f77b4",
    "flip": "#d62728",
}


# =============================================================================
# Colormaps de heatmap
#   - correlação em [-1, 1] (spearman/pearson/linear) → divergente
#   - dcor em [0, 1]                                   → sequencial
# =============================================================================

HEATMAP_DIVERGING = "bwr"  # correlação assinada [-1, 1]
HEATMAP_SEQUENTIAL = "viridis"  # dcor / magnitude [0, 1]


def heatmap_cmap(method: str) -> str:
    """Colormap do heatmap conforme o método de correlação."""
    return HEATMAP_SEQUENTIAL if method == "dcor" else HEATMAP_DIVERGING
