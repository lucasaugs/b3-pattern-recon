"""
Página: similaridade / ranking top-N (camada similarity/ de `correlations.py`).

Expõe `similar_assets` (ranking instantâneo sobre o snapshot do tensor rolling) e
os plots top-N: mais parecidos, mais opostos e mais neutros em relação a um alvo.
"""

import streamlit as st

import correlations as co
from app_common import (
    asset_select,
    get_data,
    is_heavy,
    lag_input,
    method_select,
    month_year_input,
    RAW_PRICE_EXPLANATION,
    run_plot,
    window_multi,
    window_single,
)


def _plot_raw_price_complement(
    master_df,
    asset_list,
    target,
    mode,
    direction,
    top_n,
    screening_window,
    method,
    ref_date,
):
    """
    Recupera os mesmos ativos do screening (`similar_assets`, com os parâmetros do
    plot acima) e plota o índice de preço bruto (base 100, reconstrução DTW "raw")
    do alvo + selecionados — complemento das curvas de correlação móvel
    ativo↔ativo. O screening é barato (slice do tensor já cacheado pelo plot
    anterior), então não há recálculo pesado aqui.
    """
    try:
        top_assets = co.similar_assets(
            master_df,
            asset_list,
            target,
            mode=mode,
            direction=direction,
            ref_date=ref_date,
            top_n=top_n,
            window_size=screening_window,
            method=method,
        )
    except Exception as exc:  # noqa: BLE001
        st.info(f"Sem séries brutas para exibir: {exc}")
        return
    st.markdown("**Séries brutas: índice de preço**")
    st.caption(RAW_PRICE_EXPLANATION)
    run_plot(lambda: co.plot_assets_raw_price(master_df, target, top_assets))


def render():
    st.title("🔗 Similaridade entre ativos")
    st.caption(
        "Ranqueia e plota os ativos mais parecidos, mais opostos ou mais neutros "
        "em relação a um alvo, a partir de um snapshot da correlação móvel."
    )
    master_df, asset_list = get_data()

    tab_ret, tab_macro, tab_rank = st.tabs(
        ["Top-N retornos", "Top-N macro", "Ranking (tabela)"]
    )

    # ---- Ranking tabular ----------------------------------------------------
    with tab_rank:
        st.caption("`similar_assets` — ranking direto, sem cálculo pesado.")
        with st.form("sim_rank"):
            target = asset_select(asset_list, "sr_asset")
            col1, col2 = st.columns(2)
            with col1:
                mode = st.selectbox(
                    "Modo",
                    ["returns", "macro"],
                    key="sr_mode",
                    help="returns: correlação ativo↔alvo. macro: distância entre "
                    "assinaturas macro (4-D).",
                )
                directions = (
                    ["positive", "negative", "neutral", "magnitude"]
                    if mode == "returns"
                    else ["positive", "negative"]
                )
                direction = st.selectbox("Direção", directions, key="sr_dir")
            with col2:
                top_n = st.number_input(
                    "Top N", min_value=1, max_value=30, value=5, step=1, key="sr_n"
                )
                window = window_single("sr_window", default=12)
            col3, col4 = st.columns(2)
            with col3:
                method = method_select("sr_method")
            with col4:
                lag = lag_input("sr_lag")
            _, ref_date = month_year_input(master_df, window, "sr_ref")
            submitted = st.form_submit_button("Ranquear", type="primary")
        if submitted:
            try:
                ranked = co.similar_assets(
                    master_df,
                    asset_list,
                    target,
                    mode=mode,
                    direction=direction,
                    ref_date=ref_date,
                    top_n=int(top_n),
                    window_size=window,
                    method=method,
                    lag=lag,
                )
                st.success(f"Top {len(ranked)} ({direction}, mode={mode}):")
                st.table({"#": list(range(1, len(ranked) + 1)), "Ativo": ranked})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Não foi possível ranquear: {exc}")

    # ---- Top-N (modo returns) -----------------------------------------------
    with tab_ret:
        st.caption(
            "**Como funciona:** na data de referência, `similar_assets` ranqueia "
            "os ativos pela correlação móvel ativo↔alvo na *janela de screening* "
            "e seleciona os top-N na direção escolhida (mais parecidos, opostos "
            "ou neutros). A trajetória desses ativos é então plotada em um "
            "subplot por *janela dos gráficos*."
        )
        with st.form("sim_plot_ret"):
            target = asset_select(asset_list, "spr_asset")
            kind = st.selectbox(
                "Relação",
                ["Mais parecidos", "Mais opostos", "Mais neutros (|ρ|≈0)"],
                key="spr_kind",
            )
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.number_input(
                    "Top N", min_value=1, max_value=15, value=3, step=1, key="spr_n"
                )
                method = method_select("spr_method")
            with col2:
                screening_window = window_single(
                    "spr_scr",
                    label="Janela de screening (seleção dos ativos)",
                    default=12,
                )
                lag = lag_input("spr_lag")
            windows = window_multi(
                "spr_windows", label="Janelas dos gráficos (um subplot por janela)"
            )
            # Limita pela maior janela em jogo (screening + plotadas) para que a
            # data seja válida em todos os subplots.
            ref_window = max([screening_window] + list(windows))
            _, ref_date = month_year_input(master_df, ref_window, "spr_ref")
            submitted = st.form_submit_button("Gerar", type="primary")
        if submitted:
            heavy = is_heavy(method)
            common = dict(
                top_n=int(top_n),
                window_sizes=windows,
                lag=lag,
                method=method,
                ref_date=ref_date,
                screening_window=screening_window,
            )
            if kind == "Mais parecidos":
                direction = "positive"
                run_plot(
                    lambda: co.plot_top_n_correlated_assets(
                        master_df, asset_list, target, mode="returns", **common
                    ),
                    heavy=heavy,
                )
            elif kind == "Mais opostos":
                direction = "negative"
                run_plot(
                    lambda: co.plot_top_n_inversely_correlated_assets(
                        master_df, asset_list, target, mode="returns", **common
                    ),
                    heavy=heavy,
                )
            else:
                direction = "neutral"
                run_plot(
                    lambda: co.plot_top_n_uncorrelated_assets(
                        master_df, asset_list, target, **common
                    ),
                    heavy=heavy,
                )
            _plot_raw_price_complement(
                master_df,
                asset_list,
                target,
                mode="returns",
                direction=direction,
                top_n=int(top_n),
                screening_window=screening_window,
                method=method,
                ref_date=ref_date,
            )

    # ---- Top-N (modo macro) -------------------------------------------------
    with tab_macro:
        st.caption(
            "**Como funciona:** na data de referência, os top-N são escolhidos "
            "pela distância euclidiana entre as *assinaturas macro* (vetor 4-D da "
            "correlação de cada ativo vs os IMecs) na janela de screening — "
            "menor distância = mais parecidos, maior = mais opostos. O gráfico "
            "mostra a correlação móvel ativo↔alvo dos selecionados ao lado do "
            "heatmap das assinaturas. Aqui só existe uma janela."
        )
        with st.form("sim_plot_macro"):
            target = asset_select(asset_list, "spm_asset")
            kind = st.selectbox(
                "Relação", ["Mais parecidos", "Mais opostos"], key="spm_kind"
            )
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.number_input(
                    "Top N", min_value=1, max_value=15, value=3, step=1, key="spm_n"
                )
                method = method_select("spm_method")
            with col2:
                screening_window = window_single("spm_scr", default=12)
                lag = lag_input("spm_lag")
            _, ref_date = month_year_input(master_df, screening_window, "spm_ref")
            submitted = st.form_submit_button("Gerar", type="primary")
        if submitted:
            heavy = is_heavy(method)
            common = dict(
                top_n=int(top_n),
                lag=lag,
                method=method,
                ref_date=ref_date,
                screening_window=screening_window,
            )
            if kind == "Mais parecidos":
                direction = "positive"
                run_plot(
                    lambda: co.plot_top_n_correlated_assets(
                        master_df, asset_list, target, mode="macro", **common
                    ),
                    heavy=heavy,
                )
            else:
                direction = "negative"
                run_plot(
                    lambda: co.plot_top_n_inversely_correlated_assets(
                        master_df, asset_list, target, mode="macro", **common
                    ),
                    heavy=heavy,
                )
            _plot_raw_price_complement(
                master_df,
                asset_list,
                target,
                mode="macro",
                direction=direction,
                top_n=int(top_n),
                screening_window=screening_window,
                method=method,
                ref_date=ref_date,
            )
