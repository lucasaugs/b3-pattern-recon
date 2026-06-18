"""
Infraestrutura do dashboard Streamlit do PortFlow Invest.

Reúne, em um só lugar, tudo que as páginas (`views/*.py`) precisam: o backend
matplotlib não-interativo, o carregamento cacheado do master_df, a captura de
figuras matplotlib geradas pelas funções de `correlations.py`/`clustering.py`
(que terminam em `plt.show()` e não retornam a figura) e os widgets de
parâmetros reaproveitados por várias telas.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# As funções de plot terminam em plt.show(); com o backend Agg isso só emitiria
# um warning e nos atrapalharia a capturar a figura. Neutralizamos plt.show para
# que a figura permaneça registrada no estado do pyplot até nós a capturarmos.
plt.show = lambda *args, **kwargs: None  # noqa: E731

import pandas as pd
import streamlit as st

import correlations as co
from correlations import MACRO_FEATURES, MACRO_TITLES

WINDOW_OPTIONS = [3, 6, 12, 24]
CORR_METHODS = ["dcor", "spearman"]
CLUSTER_BACKENDS = ["kmeans", "hierarchical", "spectral"]

RAW_PRICE_EXPLANATION = (
    "ℹ️ *Índice reconstruído:* Trajetória do preço em reais reconstruída a partir"
    "dos retornos mensais. Ela é calculada encadeando-os por "
    "produto acumulado partindo de uma base 100 — cada ativo é rebaseado em 100 "
    "no seu primeiro mês válido (painel desbalanceado). Logo só importa o "
    "**formato/variação relativa** da curva, não o nível absoluto; é a mesma "
    "reconstrução usada na clusterização DTW *raw*."
)


# =============================================================================
# Dados
# =============================================================================


@st.cache_data(show_spinner="Carregando master_df…")
def load_data():
    """Carrega (e cacheia) o master_df e a lista limpa de ativos."""
    master_df, asset_list = co.load_or_process_master_df()
    return master_df, sorted(asset_list)


def get_data():
    """Atalho usado por todas as páginas."""
    return load_data()


# =============================================================================
# Captura de figuras matplotlib
# =============================================================================


def show_current_fig():
    """Renderiza no Streamlit a figura matplotlib corrente e a fecha."""
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close("all")


def run_plot(plot_callable, heavy: bool = False):
    """
    Executa uma função de plot (que desenha e chama plt.show internamente) e
    publica a figura no Streamlit. `heavy=True` mostra um spinner — usado quando
    o método é `dcor` (cálculo O(N²) lento se o cache não existir).
    """
    plt.close("all")
    if heavy:
        with st.spinner("Calculando… (dcor sem cache pode levar minutos)"):
            plot_callable()
    else:
        plot_callable()
    show_current_fig()


# =============================================================================
# Widgets de parâmetros reaproveitáveis (1 widget ↔ 1 parâmetro da função)
# =============================================================================


def asset_select(asset_list, key, label="Ativo-alvo", default=None):
    index = asset_list.index(default) if default in asset_list else 0
    return st.selectbox(label, asset_list, index=index, key=key)


def asset_multiselect(asset_list, key, label="Ativos", default=None):
    return st.multiselect(label, asset_list, default=default or [], key=key)


def method_select(key, label="Método de correlação"):
    return st.selectbox(
        label,
        CORR_METHODS,
        key=key,
        help="spearman (rank, default do projeto), pearson (linear) ou "
        "dcor (dependência não-linear; LENTO se o cache não existir).",
    )


def macro_select(key, label="Indicador macro"):
    return st.selectbox(
        label, MACRO_FEATURES, format_func=lambda f: MACRO_TITLES.get(f, f), key=key
    )


def macro_multiselect(key, label="Indicadores macro", default=None):
    return st.multiselect(
        label,
        MACRO_FEATURES,
        default=MACRO_FEATURES if default is None else default,
        format_func=lambda f: MACRO_TITLES.get(f, f),
        key=key,
    )


def window_single(key, label="Janela (meses)", default=6):
    return st.selectbox(
        label, WINDOW_OPTIONS, index=WINDOW_OPTIONS.index(default), key=key
    )


def window_multi(key, label="Janelas (meses)", default=[12, 24]):
    return st.multiselect(
        label, WINDOW_OPTIONS, default=default or WINDOW_OPTIONS, key=key
    )


def lag_input(key, label="Lag (meses)"):
    return st.number_input(label, min_value=0, max_value=24, value=0, step=1, key=key)


def n_clusters_input(key, label="Nº de clusters (k)", default=4):
    return st.number_input(
        label, min_value=2, max_value=12, value=default, step=1, key=key
    )


def backend_select(key, label="Backend de clusterização"):
    return st.selectbox(label, CLUSTER_BACKENDS, key=key)


def ref_date_input(master_df, key, label="Data de referência"):
    """
    Seletor opcional de ref_date. Retorna um pd.Timestamp ou None (= última data
    disponível, semântica nativa das funções).
    """
    use_last = st.checkbox("Usar a data mais recente", value=True, key=f"{key}_last")
    if use_last:
        return None
    dates = master_df.index
    chosen = st.date_input(
        label,
        value=dates.max().date(),
        min_value=dates.min().date(),
        max_value=dates.max().date(),
        key=f"{key}_date",
    )
    return pd.Timestamp(chosen)


_MONTHS_PT = [
    "Jan",
    "Fev",
    "Mar",
    "Abr",
    "Mai",
    "Jun",
    "Jul",
    "Ago",
    "Set",
    "Out",
    "Nov",
    "Dez",
]


def month_year_input(master_df, window_size, key):
    """
    Seletor de mês/ano da data de referência, em dois inputs (ano e mês).

    """
    dates = master_df.index.sort_values()
    # O rolling só produz valor após `window_size` meses de histórico; o início é
    # descartado para que datas inválidas não apareçam (nem gerem erro).
    valid = dates[window_size:] if len(dates) > window_size else dates
    last_date = valid.max()
    first_date = valid.min()
    years = sorted({d.year for d in valid})

    col_y, col_m = st.columns(2)
    with col_y:
        year = st.selectbox("Ano", years, index=len(years) - 1, key=f"{key}_year")
    with col_m:
        month = st.selectbox(
            "Mês",
            list(range(1, 13)),
            index=last_date.month - 1,
            format_func=lambda m: _MONTHS_PT[m - 1],
            key=f"{key}_month",
        )

    target = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    target = min(max(target, first_date), last_date)

    # Snap para o fim de mês válido <= alvo (mesma lógica do snapshot helper).
    candidates = valid[valid <= target]
    resolved = candidates[-1] if len(candidates) else first_date
    if (resolved.year, resolved.month) != (year, month):
        st.caption(
            f"Sem dado para {_MONTHS_PT[month - 1]}/{year} nesta janela; "
            f"usando {_MONTHS_PT[resolved.month - 1]}/{resolved.year}."
        )

    months_ago = (last_date.year - resolved.year) * 12 + (
        last_date.month - resolved.month
    )
    return months_ago, resolved


def lookback_input(key, label="Lookback (meses, 0 = todo o histórico)", value=0):
    """Retorna um int de lookback ou None (= histórico inteiro) quando 0."""
    value = st.number_input(
        label, min_value=0, max_value=240, value=value, step=1, key=key
    )
    return None if value == 0 else int(value)


def is_heavy(method: str) -> bool:
    """dcor é o único método pesado quando o cache ainda não existe."""
    return method == "dcor"
