"""
Página: Visão Geral do Projeto (Landing Page do Dashboard).
"""

import streamlit as st

import styles


def _imec_callout(color: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="background-color:{color}1a;
                    border-left:0.4rem solid {color};
                    border-radius:0.4rem;
                    padding:0.75rem 1rem;
                    margin-bottom:0.75rem;
                    min-height:7rem;">
            <strong style="color:{color};">{title}</strong><br>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    # ---- Cabeçalho e Introdução ---------------------------------------------
    st.title("Bem-vindo ao PortFlow Invest 📈")
    st.markdown("""
        O **PortFlow Invest** é um sistema de suporte à decisão focado na diversificação 
        de portfólios no mercado financeiro brasileiro (B3), integrando dinâmicas 
        macroeconômicas locais e aprendizado de máquina não supervisionado.
        
        Em vez de analisar apenas retornos e matrizes de risco estáticas, nosso sistema 
        permite avaliar a **sensibilidade dos ativos a ciclos econômicos** e agrupar 
        comportamentos semelhantes diretamente pelas suas **trajetórias temporais brutas**.
        """)
    st.divider()

    # ---- Dados e Caracterização (POC I) -------------------------------------
    st.header("1. Robustez e qualidade dos dados")
    st.markdown(
        "A fundação das nossas análises é uma base de séries temporais exaustivamente "
        "tratada, cobrindo ativos da B3 de 2013 a 2025."
    )

    # Métricas consolidadas na caracterização estatística (characterization.ipynb)
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Registros processados", value="+7.1 Milhões")
    col2.metric(label="Inconsistências Tratadas", value="Apenas 4")
    col3.metric(label="Volume Zero Filtrado", value="170 candles")

    st.info(
        "💡As etapas de validação limparam "
        "anomalias de mercado clássicas (ruídos de liquidez e candles corrompidos). "
        "A alta integridade da base garante que os algoritmos de correlação e "
        "alinhamento temporal operem sem distorções artificiais."
    )
    st.divider()

    # ---- O Pipeline Analítico Atualizado (POC II) ---------------------------
    st.header("2. Arquitetura analítica do sistema")
    st.markdown(
        "O processamento do sistema foi desenhado para isolar e cruzar duas visões complementares do mercado:"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔗 Motor de correlação dinâmica")
        st.markdown("""
            Avalia o comportamento de ativos individuais e setoriais contra os vetores da economia. 
            Para capturar quebras de regime e respostas tardias, aplicamos:
            - **Janelas deslizantes:** Identificam como a dependência estatística muda ao longo dos anos.
            - **Lags:** Mapeiam o tempo de repasse da política econômica para os preços.
            - **Métricas utilizadas:** Spearman (para relações lineares robustas a outliers) e **Distance Correlation (dcor)** (para dependências estritamente não lineares).
            """)
    with c2:
        st.subheader("🪢 Clusterização Temporal via DTW")
        st.markdown("""
            Para identificar padrões sistêmicos de comportamento sem o viés de extração de proxies, 
            o agrupamento é feito diretamente nas trajetórias de valores brutos.
            - **Dynamic Time Warping (DTW):** Uma métrica de distância baseada em forma, capaz de alinhar séries temporais mesmo que haja aceleração, desaceleração ou deslocamento de fase nos movimentos dos ativos.
            - **Preservação temporal:** Garante que o ordenamento do tempo seja respeitado, agrupando ativos com ciclos de resiliência e queda análogos.
            """)
    st.divider()

    # ---- Os quatro IMecs ----------------------------------------------------
    st.header("3. Os 4 Indicadores Macroeconômicos (IMecs)")
    st.markdown(
        "As análises de sensibilidade individual utilizam 4 pilares fundamentais da macroeconomia brasileira, "
        "cujos comportamentos estatísticos (multimodais e de cauda pesada) justificam uma abordagem dinâmica:"
    )

    m1, m2 = st.columns(2)
    with m1:
        _imec_callout(
            styles.macro_color("macro_selic_change"),
            "🏦 Variação da Selic (%)",
            "Fator estrutural de regime de juros e custo de oportunidade.",
        )
        _imec_callout(
            styles.macro_color("macro_ipca"),
            "📈 Nível do IPCA (p.p.)",
            "Indicador inflacionário; o impacto ocorre majoritariamente através de choques e surpresas de mercado.",
        )
    with m2:
        _imec_callout(
            styles.macro_color("macro_dollar_var"),
            "💵 Variação do Dólar (%)",
            "Sensibilidade cambial, exposição a mercados globais e canais de exportação/importação.",
        )
        _imec_callout(
            styles.macro_color("macro_ibcbr_var"),
            "🏭 Variação do IBC-Br (%)",
            "Proxy mensal da atividade econômica e do crescimento real da produção.",
        )

    st.divider()

    # ---- Guia de navegação --------------------------------------------------
    st.header("4. Navegação pelo dashboard")
    st.markdown("""
        Utilize o menu lateral para explorar as capacidades da aplicação:
        - 📉 **Métricas e Correlações:** Monitore a evolução temporal da dependência entre ativos e indicadores, comparando métricas (Spearman vs Dcor) e aplicando *lags*.
        - 🧬 **Análise de Sensibilidade Macro:** Estude detalhadamente como um ativo alvo específico reage individualmente a cada um dos IMecs ao longo do tempo.
        - 🪢 **Clusterização Temporal (DTW):** Visualize o agrupamento de ativos gerado com base na proximidade de suas trajetórias temporais (*raw*), identificando setores que compartilham a mesma dinâmica histórica de movimentação.
        """)
