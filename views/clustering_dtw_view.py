"""
Página: clusterização temporal (`clustering.py`).

Agrupa ativos pelo *formato* da trajetória temporal, com dois algoritmos
shape-based da taxonomia Paparrizos:

- **DTW** (Dynamic Time Warping) — alinha séries defasadas/esticadas no tempo.
- **KShape** — correlação cruzada normalizada, invariante a deslocamento de fase.

Cada algoritmo opera sobre dois objetos: a trajetória de retorno/preço do ativo
(`cluster_assets_dtw` / `cluster_assets_kshape`) ou a trajetória da correlação
móvel contra um IMec-guia (`cluster_assets_dtw_signature` /
`cluster_assets_kshape_signature`). DTW aceita recorte por `lookback_months`;
KShape exige séries de mesmo comprimento, então usa um `series_length` fixo.
"""

import streamlit as st

import clustering as cl
from app_common import (
    get_data,
    is_heavy,
    lag_input,
    lookback_input,
    macro_select,
    method_select,
    month_year_input,
    n_clusters_input,
    RAW_PRICE_EXPLANATION,
    run_plot,
    window_single,
)

_NORMALIZATIONS = ["meanvariance", "zscore", "none"]
_METRICS = ["dtw", "softdtw"]


def _silhouette_help(dist_label: str = "DTW") -> str:
    return (
        "Mede a qualidade da clusterização. Para cada ativo, compara a distância "
        f"({dist_label}) média aos ativos do próprio cluster com a distância ao "
        "cluster vizinho mais próximo, e tira a média de todos. Varia de -1 a 1: "
        "perto de 1 = clusters coesos e bem separados; perto de 0 = clusters "
        "sobrepostos; negativo = ativos provavelmente no cluster errado."
    )


def _quality_explanation(dist_label: str = "DTW", algo_label: str = "DTW") -> str:
    return (
        "**Sobre o painel de qualidade (última linha):** cada barra horizontal é o "
        "*silhouette* de um ativo, agrupado e colorido por cluster; a linha "
        "tracejada vermelha marca o silhouette médio. O silhouette de um ativo "
        f"compara a distância ({dist_label}) média dele aos colegas do próprio "
        "cluster com a distância ao cluster vizinho mais próximo, numa escala de "
        "-1 a 1. Barras longas e positivas indicam ativos bem encaixados no seu "
        "cluster; barras curtas (perto de 0) indicam ativos na fronteira entre "
        "grupos; barras negativas sugerem ativos que estariam melhor em outro "
        "cluster. Quanto mais alto o silhouette médio, mais nítida é a separação "
        f"encontrada pelo {algo_label}."
    )


_KSHAPE_LENGTH_HELP = (
    "O KShape baseia-se em correlação cruzada (FFT) e exige séries de mesmo "
    "comprimento. Cada ativo usa seus últimos N pontos; ativos com histórico "
    "menor que N são excluídos. Maior N = formato mais detalhado, porém menos "
    "ativos elegíveis."
)

_KSHAPE_NORM_HELP = (
    "O KShape pressupõe séries escaladas; meanvariance (z-norma cada série) é a "
    "recomendação do tslearn. 'none' preserva a escala original."
)


def _guard_min_periods(lookback, min_periods, label):
    """
    Failsafe: o recorte por `lookback_months` deixa ~`lookback` pontos por ativo,
    e cada um precisa de >= `min_periods` para entrar na clusterização. Se o
    lookback for menor que esse mínimo, todos os ativos seriam descartados e a
    função levantaria erro. Aqui reduzimos o mínimo efetivo ao tamanho do
    lookback (mantendo a escolha do usuário) e avisamos no frontend qual valor
    foi usado. Retorna o `min_periods` efetivo.
    """
    if lookback is not None and lookback < min_periods:
        st.warning(
            f"O lookback ({lookback} meses) é menor que o {label} "
            f"({min_periods}). Para evitar erro, o {label} foi reduzido para "
            f"{lookback}."
        )
        return lookback
    return min_periods


# =============================================================================
# DTW
# =============================================================================


def _dtw_series_form(master_df, asset_list):
    """DTW sobre a trajetória de retorno/preço do ativo."""
    with st.form("dtw_series"):
        col1, col2, col3 = st.columns(3)
        with col1:
            k = n_clusters_input("ds_k")
            series_mode = st.selectbox(
                "Série",
                ["raw", "returns"],
                key="ds_mode",
                help="returns: retorno mensal. raw: índice de preço reconstruído.",
            )
        with col2:
            normalization = st.selectbox("Normalização", _NORMALIZATIONS, key="ds_norm")
            metric = st.selectbox("Métrica", _METRICS, key="ds_metric")
        with col3:
            min_periods = st.number_input("Mín. observações", 4, 120, 24, key="ds_minp")
            lookback = lookback_input("ds_lookback")
        # Sem janela rolling aqui: a série é o retorno/preço bruto, então
        # qualquer mês é um corte final válido (min_periods filtra os curtos).
        _, ref_date = month_year_input(master_df, 0, "ds_ref")
        submitted = st.form_submit_button("Clusterizar", type="primary")
    if submitted:
        try:
            eff_min_periods = _guard_min_periods(
                lookback, int(min_periods), "mínimo de observações"
            )
            with st.spinner("Rodando DTW…"):
                result = cl.cluster_assets_dtw(
                    master_df,
                    asset_list,
                    n_clusters=int(k),
                    series_mode=series_mode,
                    min_periods=eff_min_periods,
                    normalization=normalization,
                    metric=metric,
                    ref_date=ref_date,
                    lookback_months=lookback,
                )
            st.metric(
                "Silhouette médio",
                f"{result['silhouette']:.3f}",
                help=_silhouette_help("DTW"),
            )
            if series_mode == "raw":
                st.caption(RAW_PRICE_EXPLANATION)
            run_plot(lambda: cl.plot_dtw_clusters(result))
            st.caption(_quality_explanation("DTW", "DTW"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha no DTW: {exc}")


def _dtw_signature_form(master_df, asset_list):
    """DTW sobre a trajetória da correlação móvel contra um IMec-guia."""
    st.caption(
        "Clusteriza a trajetória de como cada ativo se correlaciona com um "
        "IMec-guia ao longo das janelas."
    )
    with st.form("dtw_sig"):
        target_macro = macro_select("dg_macro", label="IMec-guia")
        col1, col2, col3 = st.columns(3)
        with col1:
            k = n_clusters_input("dg_k")
            window = window_single("dg_window")
        with col2:
            normalization = st.selectbox(
                "Normalização", ["meanvariance", "none", "zscore"], key="dg_norm"
            )
            metric = st.selectbox("Métrica", _METRICS, key="dg_metric")
        with col3:
            method = method_select("dg_method")
            lag = lag_input("dg_lag")
        col4, col5 = st.columns(2)
        with col4:
            min_periods = st.number_input("Mín. janelas", 4, 120, 12, key="dg_minp")
            lookback = lookback_input("dg_lookback", value=12)
        with col5:
            _, ref_date = month_year_input(master_df, window, "dg_ref")
        submitted = st.form_submit_button("Clusterizar", type="primary")
    if submitted:
        try:
            eff_min_periods = _guard_min_periods(
                lookback, int(min_periods), "mínimo de janelas"
            )
            with st.spinner("Rodando DTW da assinatura…"):
                result = cl.cluster_assets_dtw_signature(
                    master_df,
                    asset_list,
                    target_macro,
                    n_clusters=int(k),
                    window_size=window,
                    lag=lag,
                    correlation_method=method,
                    min_periods=eff_min_periods,
                    normalization=normalization,
                    metric=metric,
                    ref_date=ref_date,
                    lookback_months=lookback,
                )
            st.metric(
                "Silhouette médio",
                f"{result['silhouette']:.3f}",
                help=_silhouette_help("DTW"),
            )
            run_plot(
                lambda: cl.plot_dtw_signature_clusters(result),
                heavy=is_heavy(method),
            )
            st.caption(_quality_explanation("DTW", "DTW"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha no DTW da assinatura: {exc}")


# =============================================================================
# KShape
# =============================================================================


def _kshape_series_form(master_df, asset_list):
    """KShape sobre a trajetória de retorno/preço do ativo."""
    with st.form("ks_series"):
        col1, col2, col3 = st.columns(3)
        with col1:
            k = n_clusters_input("kss_k")
            series_mode = st.selectbox(
                "Série",
                ["raw", "returns"],
                key="kss_mode",
                help="returns: retorno mensal. raw: índice de preço reconstruído.",
            )
        with col2:
            normalization = st.selectbox(
                "Normalização", _NORMALIZATIONS, key="kss_norm", help=_KSHAPE_NORM_HELP
            )
            series_length = st.number_input(
                "Comprimento da série (meses)",
                6,
                240,
                36,
                key="kss_len",
                help=_KSHAPE_LENGTH_HELP,
            )
        with col3:
            _, ref_date = month_year_input(master_df, 0, "kss_ref")
        submitted = st.form_submit_button("Clusterizar", type="primary")
    if submitted:
        try:
            with st.spinner("Rodando KShape…"):
                result = cl.cluster_assets_kshape(
                    master_df,
                    asset_list,
                    n_clusters=int(k),
                    series_mode=series_mode,
                    series_length=int(series_length),
                    normalization=normalization,
                    ref_date=ref_date,
                )
            st.metric(
                "Silhouette médio (SBD)",
                f"{result['silhouette']:.3f}",
                help=_silhouette_help("SBD"),
            )
            st.caption(
                f"{len(result['labels'])} ativos clusterizados "
                f"(com ≥ {int(series_length)} meses de histórico)."
            )
            if series_mode == "raw":
                st.caption(RAW_PRICE_EXPLANATION)
            run_plot(lambda: cl.plot_kshape_clusters(result))
            st.caption(_quality_explanation("SBD", "KShape"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha no KShape: {exc}")


def _kshape_signature_form(master_df, asset_list):
    """KShape sobre a trajetória da correlação móvel contra um IMec-guia."""
    st.caption(
        "Clusteriza, via correlação cruzada, a trajetória de como cada ativo se "
        "correlaciona com um IMec-guia ao longo das janelas."
    )
    with st.form("ks_sig"):
        target_macro = macro_select("ksg_macro", label="IMec-guia")
        col1, col2, col3 = st.columns(3)
        with col1:
            k = n_clusters_input("ksg_k")
            window = window_single("ksg_window")
        with col2:
            normalization = st.selectbox(
                "Normalização",
                ["meanvariance", "none", "zscore"],
                key="ksg_norm",
                help=_KSHAPE_NORM_HELP,
            )
            method = method_select("ksg_method")
        with col3:
            series_length = st.number_input(
                "Comprimento da série (janelas)",
                6,
                240,
                24,
                key="ksg_len",
                help=_KSHAPE_LENGTH_HELP,
            )
            lag = lag_input("ksg_lag")
        _, ref_date = month_year_input(master_df, window, "ksg_ref")
        submitted = st.form_submit_button("Clusterizar", type="primary")
    if submitted:
        try:
            spinner_msg = "Rodando KShape da assinatura…"
            if is_heavy(method):
                spinner_msg += " (dcor sem cache pode levar minutos)"
            with st.spinner(spinner_msg):
                result = cl.cluster_assets_kshape_signature(
                    master_df,
                    asset_list,
                    target_macro,
                    n_clusters=int(k),
                    window_size=window,
                    lag=lag,
                    correlation_method=method,
                    series_length=int(series_length),
                    normalization=normalization,
                    ref_date=ref_date,
                )
            st.metric(
                "Silhouette médio (SBD)",
                f"{result['silhouette']:.3f}",
                help=_silhouette_help("SBD"),
            )
            st.caption(
                f"{len(result['labels'])} ativos clusterizados "
                f"(com ≥ {int(series_length)} janelas válidas)."
            )
            run_plot(lambda: cl.plot_kshape_signature_clusters(result))
            st.caption(_quality_explanation("SBD", "KShape"))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha no KShape da assinatura: {exc}")


def render():
    st.title("🪢 Clusterização temporal")
    st.caption(
        "Agrupa ativos pelo *formato* da trajetória temporal. Escolha o algoritmo "
        "shape-based: **DTW** alinha séries defasadas/esticadas no tempo; "
        "**KShape** usa correlação cruzada, invariante a deslocamento de fase."
    )
    master_df, asset_list = get_data()

    algo = st.radio(
        "Algoritmo",
        ["DTW", "KShape"],
        horizontal=True,
        key="temporal_algo",
        help="DTW: Dynamic Time Warping (tslearn.TimeSeriesKMeans). "
        "KShape: clustering por correlação cruzada (Paparrizos & Gravano, 2015).",
    )

    tab_series, tab_sig = st.tabs(
        ["Trajetória de retorno/preço", "Trajetória da assinatura macro"]
    )
    with tab_series:
        if algo == "DTW":
            _dtw_series_form(master_df, asset_list)
        else:
            _kshape_series_form(master_df, asset_list)
    with tab_sig:
        if algo == "DTW":
            _dtw_signature_form(master_df, asset_list)
        else:
            _kshape_signature_form(master_df, asset_list)
