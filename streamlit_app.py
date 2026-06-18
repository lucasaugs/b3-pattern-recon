"""
Dashboard Streamlit do PortFlow Invest — ponto de entrada.
"""

import streamlit as st

from views import (
    clustering_dtw_view,
    clustering_macro_view,
    correlations_view,
    overview_view,
    signature_view,
    similarity_view,
)

st.set_page_config(
    page_title="PortFlow Invest",
    page_icon="📊",
    layout="wide",
)

PAGES = [
    st.Page(
        overview_view.render,
        title="Visão geral",
        icon="📊",
        url_path="visao-geral",
        default=True,
    ),
    st.Page(
        correlations_view.render,
        title="Correlações móveis",
        icon="📈",
        url_path="correlacoes",
    ),
    st.Page(
        signature_view.render,
        title="Assinatura macro",
        icon="🧬",
        url_path="assinatura",
    ),
    st.Page(
        similarity_view.render,
        title="Similaridade entre ativos",
        icon="🔗",
        url_path="similaridade",
    ),
    # st.Page(
    #     clustering_macro_view.render,
    #     title="Clusterização macro",
    #     icon="🧩",
    #     url_path="clusterizacao-macro",
    # ),
    st.Page(
        clustering_dtw_view.render,
        title="Clusterização temporal",
        icon="🪢",
        url_path="clusterizacao-temporal",
    ),
]

navigation = st.navigation(PAGES)
navigation.run()
