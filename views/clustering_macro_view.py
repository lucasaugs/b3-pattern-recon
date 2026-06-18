"""
Página: clusterização por sensibilidade macro (`clustering.py`).

Expõe `cluster_assets_per_window` (clusterização por janela com canonização de
rótulos, flip rate e silhouette), `cluster_assets_full_period` (estática) e o
`sweep_k`, com as respectivas visualizações.
"""

import streamlit as st

import clustering as cl
from app_common import (
    MACRO_TITLES,
    backend_select,
    get_data,
    is_heavy,
    lag_input,
    lookback_input,
    macro_multiselect,
    method_select,
    n_clusters_input,
    ref_date_input,
    run_plot,
    window_single,
)


def _show_metrics(result):
    col1, col2 = st.columns(2)
    col1.metric("Silhouette médio", f"{result['sil_mean']:.3f}")
    col2.metric("Flip rate médio", f"{result['flip_rate_mean']:.3f}")


def _show_membership(result):
    modal = result["modal_label"].dropna().astype(int)
    df = modal.rename("cluster").reset_index()
    df.columns = ["Ativo", "Cluster"]
    with st.expander(f"Composição dos clusters ({len(df)} ativos)"):
        st.dataframe(
            df.sort_values(["Cluster", "Ativo"]), use_container_width=True,
            hide_index=True,
        )


def render():
    st.title("🧩 Clusterização macro")
    st.caption(
        "Agrupa ativos pela sensibilidade dinâmica aos IMecs. Por janela "
        "(com flip rate) ou no período inteiro, e a varredura de k."
    )
    master_df, asset_list = get_data()

    tab_window, tab_full, tab_sweep = st.tabs(
        ["Por janela", "Período inteiro", "Sweep de k"]
    )

    # ---- Por janela ---------------------------------------------------------
    with tab_window:
        with st.form("clu_window"):
            macros = macro_multiselect("cw_macros")
            col1, col2, col3 = st.columns(3)
            with col1:
                k = n_clusters_input("cw_k")
                window = window_single("cw_window")
            with col2:
                backend = backend_select("cw_backend")
                method = method_select("cw_method")
            with col3:
                lag = lag_input("cw_lag")
                lookback = lookback_input("cw_lookback")
            ref_date = ref_date_input(master_df, "cw_ref")
            submitted = st.form_submit_button("Clusterizar", type="primary")
        if submitted:
            if len(macros) < 1:
                st.warning("Selecione ao menos um indicador macro.")
            else:
                try:
                    with st.spinner("Clusterizando por janela…"):
                        result = cl.cluster_assets_per_window(
                            master_df, asset_list, macros, n_clusters=int(k),
                            window_size=window, lag=lag, ref_date=ref_date,
                            lookback_months=lookback, cluster_method=backend,
                            correlation_method=method,
                        )
                    _show_metrics(result)
                    run_plot(lambda: cl.plot_clustering_overview(result, macros))
                    if len(macros) >= 2:
                        run_plot(lambda: cl.plot_clusters_pairs(result))
                    if len(macros) == 3:
                        run_plot(lambda: cl.plot_clusters_3d(result))
                    elif len(macros) != 3:
                        st.info(
                            "Selecione exatamente 3 indicadores para habilitar o "
                            "scatter 3D."
                        )
                    _show_membership(result)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Falha na clusterização: {exc}")

    # ---- Período inteiro ----------------------------------------------------
    with tab_full:
        st.caption("Clusterização estática sobre a correlação do período inteiro.")
        with st.form("clu_full"):
            macros = macro_multiselect("cf_macros")
            col1, col2, col3 = st.columns(3)
            with col1:
                k = n_clusters_input("cf_k")
            with col2:
                backend = backend_select("cf_backend")
            with col3:
                method = method_select("cf_method")
            lag = lag_input("cf_lag")
            submitted = st.form_submit_button("Clusterizar", type="primary")
        if submitted:
            try:
                with st.spinner("Clusterizando período inteiro…"):
                    result = cl.cluster_assets_full_period(
                        master_df, asset_list, macros, n_clusters=int(k),
                        lag=lag, cluster_method=backend, correlation_method=method,
                    )
                st.metric("Silhouette", f"{result['sil_mean']:.3f}")
                st.markdown("**Centroides (cluster × IMec)**")
                centroids = result["centroids"].rename(columns=MACRO_TITLES)
                st.dataframe(centroids.style.format("{:.2f}"), use_container_width=True)
                if len(macros) >= 2:
                    run_plot(lambda: cl.plot_clusters_pairs(result))
                if len(macros) == 3:
                    run_plot(lambda: cl.plot_clusters_3d(result))
                _show_membership(result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha na clusterização: {exc}")

    # ---- Sweep de k ---------------------------------------------------------
    with tab_sweep:
        st.caption("Varre k e plota silhouette vs flip rate.")
        with st.form("clu_sweep"):
            macros = macro_multiselect("cs_macros")
            col1, col2 = st.columns(2)
            with col1:
                k_min = st.number_input("k mínimo", 2, 11, 2, key="cs_kmin")
                window = window_single("cs_window")
                method = method_select("cs_method")
            with col2:
                k_max = st.number_input("k máximo", 3, 12, 6, key="cs_kmax")
                backend = backend_select("cs_backend")
                lag = lag_input("cs_lag")
            lookback = lookback_input("cs_lookback")
            ref_date = ref_date_input(master_df, "cs_ref")
            submitted = st.form_submit_button("Rodar sweep", type="primary")
        if submitted:
            if k_max <= k_min:
                st.warning("k máximo deve ser maior que k mínimo.")
            else:
                try:
                    with st.spinner("Varrendo k…"):
                        sweep_df = cl.sweep_k(
                            master_df, asset_list, macros,
                            k_values=range(int(k_min), int(k_max) + 1),
                            window_size=window, lag=lag, ref_date=ref_date,
                            lookback_months=lookback, cluster_method=backend,
                            correlation_method=method,
                        )
                    run_plot(lambda: cl.plot_k_sweep(sweep_df))
                    st.dataframe(
                        sweep_df.style.format("{:.3f}"), use_container_width=True
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Falha no sweep: {exc}")
