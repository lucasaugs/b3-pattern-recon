"""
Página: assinatura macro (camada signatures/ + viz/ de `correlations.py`).
"""

import streamlit as st

import correlations as co
from app_common import (
    asset_multiselect,
    asset_select,
    get_data,
    is_heavy,
    lag_input,
    macro_multiselect,
    method_select,
    month_year_input,
    run_plot,
    window_single,
)


def render():
    st.title("🧬 Assinatura macro")
    st.caption(
        "Snapshot (ativo × IMec) da correlação móvel — o descritor que a "
        "clusterização macro consome. Inspecione valores e dispersão."
    )
    master_df, asset_list = get_data()

    tab_heat, tab_dist, tab_vec = st.tabs(
        ["Heatmap", "Distribuição", "Assinatura de um ativo"]
    )

    # ---- Heatmap / distribuição / histogramas compartilham o snapshot -------
    with tab_heat:
        with st.form("sig_heat"):
            assets = asset_multiselect(
                asset_list,
                "sh_assets",
                label="Ativos (vazio = top 10 com mais histórico)",
            )
            macros = macro_multiselect("sh_macros")
            col1, col2, col3 = st.columns(3)
            with col1:
                window = window_single("sh_window")
            with col2:
                method = method_select("sh_method")
            with col3:
                lag = lag_input("sh_lag")
            months_ago, _ = month_year_input(master_df, window, "sh_months")
            submitted = st.form_submit_button("Gerar heatmap", type="primary")
        if submitted:
            run_plot(
                lambda: co.plot_asset_macro_signature_heatmap(
                    master_df,
                    asset_list=assets or None,
                    months_ago=months_ago,
                    macro_features=macros or None,
                    window_size=window,
                    method=method,
                    lag=lag,
                ),
                heavy=is_heavy(method),
            )

    with tab_dist:
        st.caption(
            "Dispersão de cada IMec entre os ativos. Colunas estreitas explicam "
            "silhouettes baixos: pouca estrutura para separar grupos."
        )
        with st.form("sig_dist"):
            assets = asset_multiselect(
                asset_list, "sd_assets", label="Ativos (vazio = todos)"
            )
            macros = macro_multiselect("sd_macros")
            col1, col2, col3 = st.columns(3)
            with col1:
                window = window_single("sd_window")
            with col2:
                method = method_select("sd_method")
            with col3:
                lag = lag_input("sd_lag")
            kind = st.radio(
                "Visualização",
                ["Boxplot", "Histogramas"],
                horizontal=True,
                key="sd_kind",
            )
            standardize = st.checkbox(
                "Padronizar (z-score por IMec)", value=True, key="sd_std"
            )
            months_ago, _ = month_year_input(master_df, window, "sd_months")
            submitted = st.form_submit_button("Gerar", type="primary")
        if submitted:
            heavy = is_heavy(method)
            if kind == "Boxplot":
                run_plot(
                    lambda: co.plot_asset_macro_signature_distribution(
                        master_df,
                        asset_list=assets or None,
                        months_ago=months_ago,
                        macro_features=macros or None,
                        window_size=window,
                        method=method,
                        lag=lag,
                        standardize=standardize,
                    ),
                    heavy=heavy,
                )
            else:
                run_plot(
                    lambda: co.plot_asset_macro_signature_histograms(
                        master_df,
                        asset_list=assets or None,
                        months_ago=months_ago,
                        macro_features=macros or None,
                        window_size=window,
                        method=method,
                        lag=lag,
                        standardize=standardize,
                    ),
                    heavy=heavy,
                )

    # ---- Vetor de assinatura de um ativo ------------------------------------
    with tab_vec:
        st.caption(
            "`macro_signature` (snapshot 4-D) e `full_period_macro_signature` "
            "(período inteiro) para um ativo."
        )
        with st.form("sig_vec"):
            target = asset_select(asset_list, "sv_asset")
            col1, col2, col3 = st.columns(3)
            with col1:
                window = window_single("sv_window")
            with col2:
                method = method_select("sv_method")
            with col3:
                lag = lag_input("sv_lag")
            _, ref_date = month_year_input(master_df, window, "sv_months")
            submitted = st.form_submit_button("Consultar", type="primary")
        if submitted:
            try:
                snap = co.macro_signature(
                    master_df,
                    asset_list,
                    target,
                    ref_date=ref_date,
                    window_size=window,
                    method=method,
                    lag=lag,
                )
                full = co.full_period_macro_signature(
                    master_df, asset_list, method=method, lag=lag
                ).loc[target]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Snapshot ({ref_date.strftime('%m/%Y')})**")
                    st.dataframe(snap.rename("correlação"))
                with col_b:
                    st.markdown("**Período inteiro**")
                    st.dataframe(full.rename("correlação"))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Não foi possível obter a assinatura: {exc}")
