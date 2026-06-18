"""
Página: correlações móveis (camada corr/ + viz/ de `correlations.py`).

Expõe as funções de visualização de rolling correlation: ativo↔IMec, ativo↔ativo
e a análise de defasagem (lag). Cada bloco é um `st.form` para que cálculos
pesados (sobretudo `dcor`) só rodem ao clicar em "Gerar".
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
    macro_select,
    method_select,
    run_plot,
    RAW_PRICE_EXPLANATION,
    window_multi,
    window_single,
)


def render():
    st.title("📈 Correlações móveis")
    st.caption(
        "Evolução temporal da correlação móvel (rolling) de um ativo contra os "
        "indicadores macro (IMecs) ou contra outros ativos, comparando janelas."
    )
    master_df, asset_list = get_data()

    tab_macro, tab_pair, tab_lag = st.tabs(
        ["Ativo × Macro", "Ativo × Ativo", "Análise de lag"]
    )

    # ---- Ativo × Macro ------------------------------------------------------
    with tab_macro:
        with st.form("rolling_macro"):
            target = asset_select(asset_list, "rm_asset")
            macros = macro_multiselect("rm_macros")
            windows = window_multi("rm_windows")
            col1, col2 = st.columns(2)
            with col1:
                method = method_select("rm_method")
            with col2:
                lag = lag_input("rm_lag")
            submitted = st.form_submit_button("Gerar", type="primary")
        if submitted:
            if not macros or not windows:
                st.warning("Selecione ao menos um indicador macro e uma janela.")
            else:
                if len(macros) == 1:
                    run_plot(
                        lambda: co.plot_rolling_windows_comparison(
                            master_df,
                            asset_list,
                            target,
                            macros[0],
                            window_sizes=windows,
                            lag=lag,
                            method=method,
                        ),
                        heavy=is_heavy(method),
                    )
                else:
                    run_plot(
                        lambda: co.plot_asset_macro_correlations_by_window(
                            master_df,
                            asset_list,
                            target,
                            macros,
                            window_sizes=windows,
                            lag=lag,
                            method=method,
                        ),
                        heavy=is_heavy(method),
                    )
                st.markdown("**Séries brutas: retorno do ativo × IMecs**")
                run_plot(
                    lambda: co.plot_asset_macro_raw_series(master_df, target, macros)
                )

    # ---- Ativo × Ativo ------------------------------------------------------
    with tab_pair:
        with st.form("rolling_pair"):
            asset_a = asset_select(asset_list, "rp_a", label="Ativo alvo")
            others = [a for a in asset_list if a != asset_a]
            asset_b = asset_multiselect(others, "rp_b", label="Comparar com")
            windows = window_multi("rp_windows")
            col1, col2 = st.columns(2)
            with col1:
                method = method_select("rp_method")
            with col2:
                lag = lag_input("rp_lag")
            submitted = st.form_submit_button("Gerar", type="primary")
        if submitted:
            if not asset_b or not windows:
                st.warning("Selecione ao menos um ativo B e uma janela.")
            else:
                run_plot(
                    lambda: co.plot_asset_pair_correlation(
                        master_df,
                        asset_list,
                        asset_a,
                        asset_b,
                        window_sizes=windows,
                        lag=lag,
                        method=method,
                    ),
                    heavy=is_heavy(method),
                )
                st.markdown("**Séries brutas: índice de preço**")
                st.caption(RAW_PRICE_EXPLANATION)
                run_plot(lambda: co.plot_assets_raw_price(master_df, asset_a, asset_b))

    # ---- Análise de lag -----------------------------------------------------
    with tab_lag:
        st.caption(
            "Compara a correlação móvel sem lag (Lag 0) contra versões defasadas, "
            "testando se o ativo reage à feature com atraso. Use as abas para "
            "escolher o tipo de feature — assim o seletor da feature atualiza "
            "junto, sem depender de um novo clique em Gerar."
        )
        lag_macro_tab, lag_asset_tab = st.tabs(["Ativo × Macro", "Ativo × Ativo"])

        # -- Lag: ativo × macro ----------------------------------------------
        with lag_macro_tab:
            with st.form("rolling_lag_macro"):
                target = asset_select(asset_list, "rlm_asset")
                feature = macro_select("rlm_feat")
                col1, col2 = st.columns(2)
                with col1:
                    window = window_single("rlm_window")
                with col2:
                    method = method_select("rlm_method")
                lags = st.multiselect(
                    "Lags a testar (meses)",
                    [1, 2, 3, 6, 12],
                    default=[1, 2],
                    key="rlm_lags",
                )
                submitted = st.form_submit_button("Gerar", type="primary")
            if submitted:
                _run_lag(
                    master_df,
                    asset_list,
                    target,
                    feature,
                    "asset-macro",
                    window,
                    lags,
                    method,
                )

        # -- Lag: ativo × ativo ----------------------------------------------
        with lag_asset_tab:
            with st.form("rolling_lag_asset"):
                target = asset_select(asset_list, "rla_asset")
                feature = asset_select(
                    [a for a in asset_list if a != target],
                    "rla_feat",
                    label="Ativo-feature",
                )
                col1, col2 = st.columns(2)
                with col1:
                    window = window_single("rla_window")
                with col2:
                    method = method_select("rla_method")
                lags = st.multiselect(
                    "Lags a testar (meses)",
                    [1, 2, 3, 6, 12],
                    default=[1, 2],
                    key="rla_lags",
                )
                submitted = st.form_submit_button("Gerar", type="primary")
            if submitted:
                _run_lag(
                    master_df,
                    asset_list,
                    target,
                    feature,
                    "asset-asset",
                    window,
                    lags,
                    method,
                )


def _run_lag(master_df, asset_list, target, feature, target_type, window, lags, method):
    """Valida os lags e dispara o plot de defasagem com o método escolhido."""
    if not lags:
        st.warning("Selecione ao menos um lag.")
        return
    run_plot(
        lambda: co.plot_feature_lag_comparison(
            master_df,
            asset_list,
            target,
            feature,
            target_type=target_type,
            window_size=window,
            lags_to_test=lags,
            method=method,
        ),
        heavy=is_heavy(method),
    )
