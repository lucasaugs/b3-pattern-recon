import hashlib
import os
import pickle

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import styles
from util_funcs import (
    load_normalize_ibc_data,
    load_normalize_ipea_data,
    treat_candle_data,
)

# =============================================================================
# Constantes
# =============================================================================

MACRO_FEATURES = [
    "macro_ipca",
    "macro_dollar_var",
    "macro_ibcbr_var",
    "macro_selic_change",
]

MACRO_TITLES = {
    "macro_selic_change": "Variação da Selic (%)",
    "macro_ipca": "Nível do IPCA (pontos percentuais)",
    "macro_dollar_var": "Variação do Dólar (%)",
    "macro_ibcbr_var": "Variação do IBC-Br (%)",
}

CANDLE_VARIABLES = ["OPEN", "HIGH", "LOW", "CLOSE"]

STATIC_CACHE_DIR = "data/cache/corr/static"
ROLLING_CACHE_DIR = "data/cache/corr/rolling"


# =============================================================================
# Camada data/ — carregamento e consolidação
# =============================================================================


def load_or_process_master_df(
    cache_path: str = "data/master_df.parquet", force: bool = False
) -> tuple:
    """
    Verifica se o cache do master_df existe. Se existir, carrega do disco.
    Caso contrário, processa os dados brutos e salva o cache em Parquet.
    Garante que o MARKET_INDEX não seja incluído na lista de ativos operacionais.
    """
    if os.path.exists(cache_path) and not force:
        master_df = pd.read_parquet(cache_path)
        forbidden_cols = MACRO_FEATURES + ["MARKET_INDEX"]
        asset_list = [c for c in master_df.columns if c not in forbidden_cols]
        return master_df, asset_list

    macro_df = get_macro_data()
    assets_dict, asset_list = get_assets_data()

    master_df = assets_dict["CLOSE"].join(macro_df, how="inner")

    master_df.to_parquet(cache_path)

    clean_asset_list = [a for a in asset_list if a != "MARKET_INDEX"]

    return master_df, clean_asset_list


def get_macro_data() -> pd.DataFrame:
    """Carrega e consolida os indicadores macroeconômicos em frequência mensal."""
    df_selic = load_normalize_ipea_data("data/ipea_selic.csv")
    df_ipca = load_normalize_ipea_data("data/ipea_ipca.csv")
    df_ipca_exp = load_normalize_ipea_data("data/ipea_exp_ipca.csv")
    df_dollar = load_normalize_ipea_data("data/ipea_camb.csv")
    df_ibc = load_normalize_ibc_data("data/ibc_br.csv")

    selic_monthly = df_selic["VALUE"].resample("ME").mean()
    macro_selic_change = selic_monthly.diff()
    macro_selic_change.name = "macro_selic_change"

    macro_ipca = df_ipca["VALUE"].resample("ME").first()
    macro_ipca.name = "macro_ipca"

    ipca_exp_monthly = df_ipca_exp["VALUE"].resample("ME").first()
    macro_ipca_surprise = macro_ipca - ipca_exp_monthly
    macro_ipca_surprise.name = "macro_ipca_surprise"

    dollar_monthly = df_dollar["VALUE"].resample("ME").last()
    macro_dollar_var = dollar_monthly.pct_change() * 100
    macro_dollar_var.name = "macro_dollar_var"

    ibc_monthly = df_ibc["IBC_HEADLINE"].resample("ME").first()
    macro_ibcbr_var = ibc_monthly.pct_change() * 100
    macro_ibcbr_var.name = "macro_ibcbr_var"

    macro_df = pd.concat(
        [
            macro_selic_change,
            macro_ipca,
            macro_dollar_var,
            macro_ibcbr_var,
        ],
        axis=1,
    )
    return macro_df


def get_assets_data(candle_path: str = "data/b3_candles_raw.csv") -> tuple:
    """
    Carrega, filtra e calcula o retorno mensal dos ativos da B3.
    Mantém o MARKET_INDEX no DataFrame para benchmark, mas não na lista de nomes.
    """
    candle_data = treat_candle_data(candle_path)

    candle_data["YEAR"] = candle_data["TIMESTAMP"].dt.year
    availability = candle_data[["NAME", "YEAR"]].drop_duplicates()
    availability["has_data"] = 1
    recent_assets = (
        availability["NAME"]
        .value_counts()[availability["NAME"].value_counts() <= 4]
        .index.tolist()
    )
    candle_data = candle_data[~candle_data["NAME"].isin(recent_assets)].copy()
    candle_data.drop(columns=["YEAR"], inplace=True)

    asset_counts = candle_data["NAME"].value_counts()
    small_assets = asset_counts[asset_counts < 10000].index.tolist()
    candle_data = candle_data[~candle_data["NAME"].isin(small_assets)].copy()

    if not pd.api.types.is_datetime64_any_dtype(candle_data["TIMESTAMP"]):
        candle_data["TIMESTAMP"] = pd.to_datetime(candle_data["TIMESTAMP"])

    assets_monthly_pct_change = {}
    for var in CANDLE_VARIABLES:
        assets_monthly_var = (
            candle_data.set_index("TIMESTAMP")
            .groupby("NAME")[var]
            .resample("ME")
            .last()
            .unstack(level=0)
        )
        assets_monthly_ret_var = assets_monthly_var.pct_change() * 100
        assets_monthly_ret_var["MARKET_INDEX"] = assets_monthly_ret_var.mean(axis=1)
        assets_monthly_pct_change[var] = assets_monthly_ret_var

    valid_asset_names = candle_data["NAME"].unique().tolist()

    return assets_monthly_pct_change, valid_asset_names


# =============================================================================
# Camada corr/ — engine de correlações + cache
# =============================================================================


def _build_lag_suffix(lag: int, lag_features: list) -> str:
    """Gera o sufixo de fingerprint do cache a partir do lag e das features defasadas."""
    if not lag or not lag_features:
        return ""
    feats_str = "_".join(lag_features)
    return f"_L{lag}_{feats_str}"


def _build_assets_hash(asset_names: list) -> str:
    """Hash curto e determinístico do conjunto de ativos para diferenciar caches."""
    canonical = ",".join(sorted(asset_names)).encode("utf-8")
    return hashlib.md5(canonical).hexdigest()[:8]


def _apply_lag(master_data: pd.DataFrame, lag: int, lag_features: list) -> pd.DataFrame:
    """Aplica `.shift(lag)` apenas às colunas indicadas, retornando uma cópia se necessário."""
    if not lag or not lag_features:
        return master_data
    cols = [c for c in lag_features if c in master_data.columns]
    if not cols:
        return master_data
    working_data = master_data.copy()
    working_data[cols] = working_data[cols].shift(lag)
    return working_data


def calculate_static_correlations(
    master_data: pd.DataFrame,
    asset_names: list,
    method: str = "spearman",
    lag: int = 0,
    lag_features: list = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Calcula a correlação estática entre os ativos e os IMecs para todo o período.
    Aplica `.shift(lag)` nas `lag_features` internamente, garantindo que o cache
    diferencie computações com defasagens distintas via fingerprint no nome do arquivo.
    """
    available_assets = [c for c in asset_names if c in master_data.columns]

    os.makedirs(STATIC_CACHE_DIR, exist_ok=True)
    lag_suffix = _build_lag_suffix(lag, lag_features)
    assets_hash = _build_assets_hash(available_assets)
    cache_filename = f"static_{method}_A{assets_hash}{lag_suffix}.parquet"
    cache_path = os.path.join(STATIC_CACHE_DIR, cache_filename)

    if os.path.exists(cache_path) and not force:
        cached_df = pd.read_parquet(cache_path)
        valid_idx = [a for a in available_assets if a in cached_df.index]
        return cached_df.loc[valid_idx, MACRO_FEATURES]

    working_data = _apply_lag(master_data, lag, lag_features)

    if method == "dcor":
        dcor_frame = pd.DataFrame(
            index=available_assets, columns=MACRO_FEATURES, dtype=float
        )
        for asset in available_assets:
            for macro in MACRO_FEATURES:
                valid_pairs = working_data[[asset, macro]].dropna()
                if len(valid_pairs) > 1:
                    dcor_frame.loc[asset, macro] = dcor.distance_correlation(
                        valid_pairs[asset].values, valid_pairs[macro].values
                    )
                else:
                    dcor_frame.loc[asset, macro] = np.nan
        result_df = dcor_frame
    else:
        corr_matrix = working_data.corr(method=method)
        result_df = corr_matrix.loc[available_assets, MACRO_FEATURES]

    result_df.to_parquet(cache_path)

    return result_df


def calculate_rolling_correlations(
    master_data: pd.DataFrame,
    asset_list: list,
    window_size: int,
    method: str = "spearman",
    target: str = "asset-macro",
    lag: int = 0,
    lag_features: list = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Calcula matrizes de correlação móveis ignorando o MARKET_INDEX.
    Aplica `.shift(lag)` nas `lag_features` internamente para que o cache reflita
    a defasagem aplicada, evitando colisões com computações sem lag (ou com lag distinto).
    """
    clean_list = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]

    os.makedirs(ROLLING_CACHE_DIR, exist_ok=True)
    lag_suffix = _build_lag_suffix(lag, lag_features)
    assets_hash = _build_assets_hash(clean_list)
    cache_filename = (
        f"rolling_W{window_size}_{method}_{target}_A{assets_hash}{lag_suffix}.pkl"
    )
    cache_path = os.path.join(ROLLING_CACHE_DIR, cache_filename)

    if os.path.exists(cache_path) and not force:
        with open(cache_path, "rb") as f:
            cached_result = pickle.load(f)

        idx = pd.IndexSlice
        target_cols = clean_list if target == "asset-asset" else MACRO_FEATURES
        # Usar os valores REALMENTE presentes no índice, não `index.levels[1]`:
        # após o dropna(how="all") (assets sem nenhuma janela válida — comum em
        # dcor) a tupla some do índice, mas o rótulo permanece como categoria em
        # `.levels[1]`, fazendo o `.loc` abaixo levantar KeyError.
        present_assets = cached_result.index.get_level_values(1).unique()
        valid_assets = [a for a in clean_list if a in present_assets]

        print("Loaded rolling correlations from cache.")
        return cached_result.loc[idx[:, valid_assets], target_cols]

    working_data = _apply_lag(master_data, lag, lag_features)

    print("Calculating rolling correlations...")
    if method == "pearson":
        rolling_corr = working_data.rolling(window=window_size).corr()
    else:
        frames = []
        keys = []
        min_periods = max(4, int(window_size * 0.5))

        for i in range(window_size - 1, len(working_data)):
            window = working_data.iloc[i - window_size + 1 : i + 1]

            if method == "dcor":
                target_cols = clean_list if target == "asset-asset" else MACRO_FEATURES
                local_frame = pd.DataFrame(
                    index=clean_list, columns=target_cols, dtype=float
                )

                for r_col in clean_list:
                    for c_col in target_cols:
                        valid_pairs = window[[r_col, c_col]].dropna()

                        if len(valid_pairs) >= min_periods:
                            local_frame.loc[r_col, c_col] = dcor.distance_correlation(
                                valid_pairs[r_col].values, valid_pairs[c_col].values
                            )

                frames.append(local_frame)
            else:
                frames.append(window.corr(method=method))

            keys.append(working_data.index[i])

        rolling_corr = pd.concat(frames, keys=keys, names=["TIMESTAMP", "Variável"])

    idx = pd.IndexSlice

    if target == "asset-macro":
        result = rolling_corr.loc[idx[:, clean_list], MACRO_FEATURES]
    elif target == "asset-asset":
        result = rolling_corr.loc[idx[:, clean_list], clean_list]
    else:
        raise ValueError("Target parameter must be 'asset-macro' or 'asset-asset'")

    # ffill POR ATIVO (groupby level=1): preserva a sensibilidade quando a macro
    # fica inerte (ex.: meses sem mudança na Selic → variância zero na janela →
    # correlação NaN). Um .ffill() plano propagaria o valor da linha de cima, que
    # é OUTRO ativo (índice é data-maior, ativo-menor), vazando valores entre
    # ativos. groupby(level=1) força o forward-fill ao longo do tempo de cada ativo.
    final_result = result.groupby(level=1).ffill().dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(final_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return final_result


# =============================================================================
# Camada signatures/ — descritores por ativo num ref_date (snapshot)
# =============================================================================


def _resolve_ref_date(ref_date, available_dates) -> pd.Timestamp:
    """Resolve ref_date contra o índice temporal; default = última data disponível."""
    if ref_date is None:
        return available_dates[-1]
    selected = pd.to_datetime(ref_date)
    if selected in available_dates:
        return selected
    earlier = available_dates[available_dates <= selected]
    if len(earlier) == 0:
        raise ValueError(
            f"ref_date {ref_date} é anterior ao início da série rolling disponível."
        )
    return earlier[-1]


def macro_signature(
    master_data: pd.DataFrame,
    asset_list: list,
    asset: str,
    ref_date=None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    lag_features: list = None,
) -> pd.Series:
    """
    Assinatura macro (snapshot 4-D) do ativo em ref_date.

    Vetor com a correlação móvel do ativo vs cada IMec, derivado por slice do tensor
    rolling_asset_macro — sem nenhum cálculo novo, só leitura do cache.
    """
    rolling = calculate_rolling_correlations(
        master_data,
        asset_list,
        window_size=window_size,
        target="asset-macro",
        method=method,
        lag=lag,
        lag_features=lag_features,
    )
    dates = rolling.index.get_level_values(0).unique()
    selected = _resolve_ref_date(ref_date, dates)
    snapshot = rolling.xs(selected, level=0)
    return snapshot.loc[asset, MACRO_FEATURES]


def full_period_macro_signature(
    master_data: pd.DataFrame,
    asset_list: list,
    method: str = "spearman",
    lag: int = 0,
    lag_features: list = None,
    exclude_features: list = None,
) -> pd.DataFrame:
    """
    Macro signature (asset x IMec) agregada sobre toda a série temporal.

    Diferente de `macro_signature`, que devolve o snapshot da rolling correlation
    em um ref_date, esta versão usa a correlação estática do período inteiro como
    descritor por ativo. Wrapper fino sobre `calculate_static_correlations` que
    filtra MARKET_INDEX e, opcionalmente, remove features.
    """
    clean_list = [a for a in asset_list if a != "MARKET_INDEX"]
    sig = calculate_static_correlations(
        master_data,
        clean_list,
        method=method,
        lag=lag,
        lag_features=lag_features,
    )
    if exclude_features:
        keep = [c for c in sig.columns if c not in exclude_features]
        sig = sig[keep]
    return sig


def returns_signature(
    master_data: pd.DataFrame,
    asset_list: list,
    asset: str,
    ref_date=None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    lag_features: list = None,
) -> pd.Series:
    """
    Assinatura de retornos (snapshot (N-1)-D) do ativo em ref_date.

    Linha do tensor rolling_asset_asset em ref_date: a correlação móvel do ativo
    contra cada um dos demais. A própria linha do alvo é descartada.
    """
    rolling = calculate_rolling_correlations(
        master_data,
        asset_list,
        window_size=window_size,
        target="asset-asset",
        method=method,
        lag=lag,
        lag_features=lag_features,
    )
    dates = rolling.index.get_level_values(0).unique()
    selected = _resolve_ref_date(ref_date, dates)
    snapshot = rolling.xs(selected, level=0)
    row = snapshot.loc[asset]
    return row.drop(asset, errors="ignore")


# =============================================================================
# Camada similarity/ — ranking top-N a partir das assinaturas
# =============================================================================


def similar_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    mode: str = "returns",
    direction: str = "positive",
    ref_date=None,
    top_n: int = 3,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    lag_features: list = None,
) -> list:
    """
    Ranqueia ativos por similaridade ao alvo no instante ref_date.

    Toda a operação é um slice + ranking sobre tensores já calculados em corr/ —
    não há cálculo pesado aqui; trocar `target_asset` ou `ref_date` é instantâneo.

    Parameters
    ----------
    mode :
        - "returns": usa o snapshot da linha do rolling_asset_asset em ref_date,
          i.e. ranqueia diretamente pela correlação móvel ativo↔alvo no mercado real.
        - "macro":   usa a distância Euclidiana entre macro_signatures (4-D) em ref_date,
          i.e. mede o quanto dois ativos reagem de forma parecida aos IMecs.
    direction :
        - "positive":  mais parecidos (maior correlação em "returns" / menor distância em "macro").
        - "negative":  mais opostos (correlação mais negativa em "returns" / maior distância em "macro").
        - "neutral":   correlação com |p| mais próxima de zero (apenas mode="returns").
        - "magnitude": maior |p| sem importar o sinal (apenas mode="returns").
    ref_date :
        Data de referência do snapshot. Se None, usa a última data disponível.
        Se a data não existir exatamente, cai para a última data ≤ ref_date.
    """
    if mode == "returns":
        ranking = returns_signature(
            master_data,
            asset_list,
            target_asset,
            ref_date=ref_date,
            window_size=window_size,
            method=method,
            lag=lag,
            lag_features=lag_features,
        ).dropna()

        if direction == "positive":
            return ranking.nlargest(top_n).index.tolist()
        if direction == "negative":
            return ranking.nsmallest(top_n).index.tolist()
        if direction == "neutral":
            return ranking.abs().nsmallest(top_n).index.tolist()
        if direction == "magnitude":
            return ranking.abs().nlargest(top_n).index.tolist()
        raise ValueError(
            f"direction inválido para mode='returns': {direction!r}. "
            "Use 'positive', 'negative', 'neutral' ou 'magnitude'."
        )

    if mode == "macro":
        rolling = calculate_rolling_correlations(
            master_data,
            asset_list,
            window_size=window_size,
            target="asset-macro",
            method=method,
            lag=lag,
            lag_features=lag_features,
        )
        dates = rolling.index.get_level_values(0).unique()
        selected = _resolve_ref_date(ref_date, dates)
        snapshot = rolling.xs(selected, level=0)[MACRO_FEATURES].dropna()

        if target_asset not in snapshot.index:
            raise ValueError(
                f"{target_asset} não possui macro_signature válida em {selected.date()}."
            )

        target_vec = snapshot.loc[target_asset].values
        diffs = snapshot.values - target_vec
        distances = pd.Series(np.linalg.norm(diffs, axis=1), index=snapshot.index).drop(
            target_asset
        )

        if direction == "positive":
            return distances.nsmallest(top_n).index.tolist()
        if direction == "negative":
            return distances.nlargest(top_n).index.tolist()
        raise ValueError(
            f"direction={direction!r} não é suportado em mode='macro'; "
            "use 'positive' ou 'negative'."
        )

    raise ValueError(f"mode inválido: {mode!r}; use 'returns' ou 'macro'.")


# =============================================================================
# Camada viz/ — apenas slice + plot (nenhum cálculo pesado aqui)
# =============================================================================


def _plot_rolling_subplots(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    features_to_plot: list,
    target_type: str,
    window_sizes: list,
    background_color: str,
    title: str,
    legend_title: str,
    lag: int = 0,
    method: str = "spearman",
    ref_date=None,
    mark_ref_date: bool = False,
) -> None:
    """Motor gráfico com eixos X alinhados pelo período de convergência real das janelas."""
    sns.set_theme(style="whitegrid")
    num_windows = len(window_sizes)

    fig, axes = plt.subplots(
        num_windows, 1, figsize=(15, 4.5 * num_windows), sharex=True
    )
    fig.patch.set_facecolor(background_color)

    if num_windows == 1:
        axes = [axes]

    lag_suffix = f" (Lag: {lag}M)" if lag > 0 else ""
    fig.suptitle(f"{title}{lag_suffix} [{method.upper()}]", fontsize=16, y=1.02)

    is_single_feature = len(features_to_plot) == 1
    # Cor por janela só faz sentido quando há UMA série (asset↔asset single):
    # para asset↔macro a cor é fixa por IMec; para multi-asset, paleta de ativos.
    use_window_colors = is_single_feature and target_type != "asset-macro"

    if use_window_colors:
        window_colors = styles.palette("windows", num_windows)
    elif target_type == "asset-macro":
        feature_colors = [styles.macro_color(f) for f in features_to_plot]
    else:
        feature_colors = styles.palette("assets", len(features_to_plot))

    calculated_series = []

    for ax_idx, window in enumerate(window_sizes):
        rolling_corr_df = calculate_rolling_correlations(
            master_data,
            asset_list,
            window_size=window,
            target=target_type,
            method=method,
            lag=lag,
            lag_features=features_to_plot,
        )
        calculated_series.append(rolling_corr_df)

    all_dates = pd.concat(
        [df.index.get_level_values(0).to_series() for df in calculated_series]
    )
    shared_min_date = all_dates.min()
    shared_max_date = all_dates.max()

    selected_date = None
    if mark_ref_date or ref_date is not None:
        union_dates = pd.DatetimeIndex(sorted(all_dates.unique()))
        selected_date = _resolve_ref_date(ref_date, union_dates)

    for ax_idx, (ax, window) in enumerate(zip(axes, window_sizes)):
        ax.set_facecolor(background_color)
        rolling_corr_df = calculated_series[ax_idx]
        ref_handles = []

        # O alvo pode não ter linha nesta janela: dcor exige min_periods=4 pares
        # válidos, então janelas curtas (ex.: W3) produzem tensor vazio e o
        # `.xs` abaixo levantaria KeyError. Degrada graciosamente com um aviso.
        present_assets = (
            rolling_corr_df.index.get_level_values(1).unique()
            if len(rolling_corr_df)
            else pd.Index([])
        )
        if target_asset not in present_assets:
            ax.text(
                0.5,
                0.5,
                f"Sem dados de correlação para {target_asset}\n"
                f"(janela de {window}M insuficiente para o método {method.upper()})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
            ax.set_title(f"Janela de {window} meses", fontsize=13)
            continue

        for i, feature in enumerate(features_to_plot):
            time_series = rolling_corr_df.xs(target_asset, level=1)[feature]

            line_color = (
                window_colors[ax_idx] if use_window_colors else feature_colors[i]
            )
            label = (
                MACRO_TITLES.get(feature, feature)
                if target_type == "asset-macro"
                else feature
            )

            ax.plot(
                time_series.index,
                time_series,
                color=line_color,
                linewidth=styles.lw("primary"),
                label=label,
            )

            if selected_date is not None and selected_date in time_series.index:
                value = time_series.loc[selected_date]
                if not pd.isna(value):
                    ax.scatter(
                        [selected_date],
                        [value],
                        color=line_color,
                        s=60,
                        zorder=5,
                        edgecolor="black",
                        linewidths=0.8,
                    )
                    handle_label = (
                        f"{value:.2f}" if is_single_feature else f"{label}: {value:.2f}"
                    )
                    ref_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            markersize=8,
                            markerfacecolor=line_color,
                            markeredgecolor="black",
                            linestyle="",
                            label=handle_label,
                        )
                    )

        if selected_date is not None:
            ax.axvline(
                selected_date,
                color=styles.COLORS["ref_line"],
                linestyle=":",
                linewidth=styles.lw("reference"),
                alpha=0.8,
            )

        if method != "dcor":
            ax.axhline(
                0,
                color=styles.COLORS["zero_line"],
                linestyle="--",
                linewidth=styles.lw("reference"),
            )

        ax.set_xlim(shared_min_date, shared_max_date)

        ax.set_title(f"Janela de {window} meses", fontsize=13)
        main_legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title=legend_title,
            borderaxespad=0.0,
        )

        if ref_handles:
            ref_str = pd.Timestamp(selected_date).strftime("%Y-%m")
            ax.legend(
                handles=ref_handles,
                loc="upper right",
                title=f"Valor em {ref_str}",
                fontsize=9,
                title_fontsize=10,
            )
            if main_legend is not None:
                ax.add_artist(main_legend)

        if method == "dcor":
            ax.set_ylabel("Distance Corr.")
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel(method.capitalize())
            ax.set_ylim(-1.1, 1.1)

    plt.xlabel("Tempo", fontsize=12)
    plt.tight_layout()
    plt.show()


_MACRO_SHORT_LABELS = {
    "macro_selic_change": "Selic",
    "macro_ipca": "IPCA",
    "macro_dollar_var": "Dólar",
    "macro_ibcbr_var": "IBC-Br",
}


def _plot_macro_subplots(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    extra_assets: list,
    window_size: int,
    background_color: str,
    title: str,
    lag: int = 0,
    method: str = "spearman",
    ref_date=None,
    mark_ref_date: bool = False,
) -> None:
    """
    Layout 2-subplots para o modo 'macro' dos top-N:
      - esquerda (~60% largura): correlação móvel asset↔asset do target vs extra_assets,
        sob `window_size`, com marcação na data de referência.
      - direita (~40% largura): heatmap das macro_signatures (asset↔IMec) de
        target + extra_assets avaliadas no `ref_date` resolvido.
    """
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(background_color)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)
    ax_lines = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])
    ax_lines.set_facecolor(background_color)
    ax_hm.set_facecolor(background_color)

    lag_suffix = f" (Lag: {lag}M)" if lag > 0 else ""
    fig.suptitle(
        f"{title}{lag_suffix} [{method.upper()}]",
        fontsize=15,
        y=1.02,
    )

    line_assets = [a for a in extra_assets if a != target_asset]
    line_colors = styles.palette("assets", len(line_assets))

    rolling_aa = calculate_rolling_correlations(
        master_data,
        asset_list,
        window_size=window_size,
        target="asset-asset",
        method=method,
        lag=lag,
        lag_features=line_assets,
    )

    all_dates = rolling_aa.index.get_level_values(0)
    selected_date = None
    if mark_ref_date or ref_date is not None:
        union_dates = pd.DatetimeIndex(sorted(all_dates.unique()))
        selected_date = _resolve_ref_date(ref_date, union_dates)

    target_slice = rolling_aa.xs(target_asset, level=1)
    ref_handles = []

    for i, asset in enumerate(line_assets):
        if asset not in target_slice.columns:
            continue
        series = target_slice[asset]
        color = line_colors[i]
        ax_lines.plot(
            series.index, series, color=color, linewidth=styles.lw("primary"), label=asset
        )

        if selected_date is not None and selected_date in series.index:
            value = series.loc[selected_date]
            if not pd.isna(value):
                ax_lines.scatter(
                    [selected_date],
                    [value],
                    color=color,
                    s=60,
                    zorder=5,
                    edgecolor="black",
                    linewidths=0.8,
                )
                ref_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        markersize=8,
                        markerfacecolor=color,
                        markeredgecolor="black",
                        linestyle="",
                        label=f"{asset}: {value:.2f}",
                    )
                )

    if selected_date is not None:
        ax_lines.axvline(
            selected_date,
            color=styles.COLORS["ref_line"],
            linestyle=":",
            linewidth=styles.lw("reference"),
            alpha=0.8,
        )
    if method != "dcor":
        ax_lines.axhline(
            0,
            color=styles.COLORS["zero_line"],
            linestyle="--",
            linewidth=styles.lw("reference"),
        )

    ax_lines.set_xlim(all_dates.min(), all_dates.max())
    ax_lines.set_title("Correlação móvel asset↔asset", fontsize=12)
    ax_lines.set_xlabel("Tempo", fontsize=11)
    if method == "dcor":
        ax_lines.set_ylabel("Distance Corr.")
        ax_lines.set_ylim(-0.05, 1.05)
    else:
        ax_lines.set_ylabel(method.capitalize())
        ax_lines.set_ylim(-1.1, 1.1)
    main_legend = ax_lines.legend(loc="upper left", title="Ativos", fontsize=9)

    if ref_handles:
        ref_str = pd.Timestamp(selected_date).strftime("%Y-%m")
        ax_lines.legend(
            handles=ref_handles,
            loc="upper right",
            title=f"Valor em {ref_str}",
            fontsize=9,
            title_fontsize=10,
        )
        ax_lines.add_artist(main_legend)

    rolling_am = calculate_rolling_correlations(
        master_data,
        asset_list,
        window_size=window_size,
        target="asset-macro",
        method=method,
        lag=lag,
        lag_features=MACRO_FEATURES,
    )

    if selected_date is None:
        am_dates = rolling_am.index.get_level_values(0).unique()
        selected_date = am_dates[-1]

    snapshot = rolling_am.xs(selected_date, level=0)
    heatmap_assets = [a for a in [target_asset] + line_assets if a in snapshot.index]
    sig_df = snapshot.loc[heatmap_assets, MACRO_FEATURES].rename(
        columns=_MACRO_SHORT_LABELS
    )

    if method == "dcor":
        hm_kwargs = dict(vmin=0, vmax=1, cmap=styles.heatmap_cmap(method))
    else:
        hm_kwargs = dict(vmin=-1, vmax=1, cmap=styles.heatmap_cmap(method), center=0)

    sns.heatmap(
        sig_df,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax_hm,
        cbar_kws={"label": "Correlação asset↔IMec"},
        **hm_kwargs,
    )
    ref_str = pd.Timestamp(selected_date).strftime("%Y-%m")
    ax_hm.set_title(f"Macro signatures em {ref_str}", fontsize=12)
    ax_hm.set_xlabel("IMec", fontsize=11)
    ax_hm.set_ylabel("Ativo", fontsize=11)
    ax_hm.tick_params(axis="x", rotation=0)
    ax_hm.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.show()


def plot_rolling_windows_comparison(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    target_macro: str,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
) -> None:
    """Gera subplots comparando a correlação móvel de um ativo vs um IMec."""
    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=[target_macro],
        target_type="asset-macro",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução da Correlação: {target_asset} vs {MACRO_TITLES.get(target_macro, target_macro)}",
        legend_title="Indicador Macro",
        lag=lag,
        method=method,
    )


def plot_asset_macro_correlations_by_window(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    macro_features: list,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
) -> None:
    """Gera subplots com a correlação móvel do ativo contra os IMecs."""
    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=macro_features,
        target_type="asset-macro",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução das Correlações Macroeconômicas para {target_asset}",
        legend_title="Indicadores Macro",
        lag=lag,
        method=method,
    )


def plot_asset_pair_correlation(
    master_data: pd.DataFrame,
    asset_list: list,
    asset_a: str,
    asset_b,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
) -> None:
    """Gera subplots comparando a evolução da correlação móvel entre pares de ativos."""
    features_to_plot = asset_b if isinstance(asset_b, list) else [asset_b]
    b_title = ", ".join(asset_b) if isinstance(asset_b, list) else asset_b

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=asset_a,
        features_to_plot=features_to_plot,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução da Correlação: {asset_a} vs {b_title}",
        legend_title="Ativo(s)",
        lag=lag,
        method=method,
    )


def _format_screening_label(mode: str, ref_date, screening_window: int) -> str:
    """Rótulo curto para os títulos dos plots top-N — descreve como o screening foi feito."""
    label_date = pd.to_datetime(ref_date).strftime("%Y-%m") if ref_date else "última"
    return f"mode={mode}, W={screening_window}M, ref={label_date}"


def plot_top_n_correlated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
    mode: str = "returns",
    ref_date=None,
    screening_window: int = 12,
) -> None:
    """
    Plota a evolução da correlação móvel dos top_n ativos mais parecidos com o alvo.

    O screening é feito por similar_assets() em ref_date — i.e. um snapshot do tensor
    rolling (não a correlação estática dos retornos brutos, como na versão anterior).
    Em mode="returns" o ranking vem direto da linha do rolling_asset_asset; em
    mode="macro" da distância Euclidiana entre macro_signatures.
    """
    top_assets = similar_assets(
        master_data,
        asset_list,
        target_asset,
        mode=mode,
        direction="positive",
        ref_date=ref_date,
        top_n=top_n,
        window_size=screening_window,
        method=method,
    )

    base_title = (
        f"Top {top_n} Mais Parecidos com {target_asset}"
        # f"screening: {_format_screening_label(mode, ref_date, screening_window)}"
    )

    if mode == "macro":
        _plot_macro_subplots(
            master_data=master_data,
            asset_list=asset_list,
            target_asset=target_asset,
            extra_assets=top_assets,
            window_size=screening_window,
            background_color=background_color,
            title=base_title,
            lag=lag,
            method=method,
            ref_date=ref_date,
            mark_ref_date=True,
        )
        return

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=top_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=base_title,
        legend_title="Ativos",
        lag=lag,
        method=method,
        ref_date=ref_date,
        mark_ref_date=True,
    )


def plot_top_n_inversely_correlated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
    mode: str = "returns",
    ref_date=None,
    screening_window: int = 12,
) -> None:
    """
    Plota a evolução da correlação móvel dos top_n ativos mais opostos ao alvo.

    Screening via similar_assets(direction="negative"): correlação mais negativa em
    mode="returns" / maior distância Euclidiana em mode="macro".
    """
    top_assets = similar_assets(
        master_data,
        asset_list,
        target_asset,
        mode=mode,
        direction="negative",
        ref_date=ref_date,
        top_n=top_n,
        window_size=screening_window,
        method=method,
    )

    base_title = (
        f"Top {top_n} Mais Opostos a {target_asset}"
        # f"screening: {_format_screening_label(mode, ref_date, screening_window)}"
    )

    if mode == "macro":
        _plot_macro_subplots(
            master_data=master_data,
            asset_list=asset_list,
            target_asset=target_asset,
            extra_assets=top_assets,
            window_size=screening_window,
            background_color=background_color,
            title=base_title,
            lag=lag,
            method=method,
            ref_date=ref_date,
            mark_ref_date=True,
        )
        return

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=top_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=base_title,
        legend_title="Ativos",
        lag=lag,
        method=method,
        ref_date=ref_date,
        mark_ref_date=True,
    )


def plot_top_n_uncorrelated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = styles.BACKGROUND,
    lag: int = 0,
    method: str = "spearman",
    ref_date=None,
    screening_window: int = 12,
) -> None:
    """
    Plota a evolução dos top_n ativos mais neutros (|ρ|≈0) com o alvo.

    Só mode="returns" — "neutralidade" não tem análogo natural em mode="macro"
    (distância 4-D é sempre não-negativa).
    """
    top_assets = similar_assets(
        master_data,
        asset_list,
        target_asset,
        mode="returns",
        direction="neutral",
        ref_date=ref_date,
        top_n=top_n,
        window_size=screening_window,
        method=method,
    )

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=top_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=(
            f"Top {top_n} Neutros com {target_asset} — "
            f"screening: {_format_screening_label('returns', ref_date, screening_window)}"
        ),
        legend_title="Ativos",
        lag=lag,
        method=method,
        ref_date=ref_date,
        mark_ref_date=True,
    )


def plot_feature_lag_comparison(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    target_feature: str,
    target_type: str = "asset-macro",
    window_size: int = 6,
    lags_to_test: list = [3, 6, 12],
    method: str = "spearman",
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Compara a evolução da correlação móvel de um ativo contra uma feature
    (macro ou ativo). Cada lag ocupa uma linha do grid: à esquerda, a série de
    correlação original (Lag 0) vs. a defasada com a diferença sombreada; à
    direita, um painel com a média ± desvio-padrão de cada série e o delta entre
    elas.

    `method` (`spearman`/`pearson`/`dcor`) é repassado às rolling correlations e
    define o eixo-y do gráfico de linhas: `[-0.05, 1.05]` para `dcor`,
    `[-1.1, 1.1]` caso contrário.
    """
    active_lags = [lag for lag in lags_to_test if lag > 0]

    if not active_lags:
        raise ValueError(
            "A lista lags_to_test deve conter pelo menos um valor maior que 0."
        )

    is_dcor = method == "dcor"
    ylim = (-0.05, 1.05) if is_dcor else (-1.1, 1.1)
    y_label = "Distance Corr." if is_dcor else method.capitalize()

    num_rows = len(active_lags)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        num_rows,
        2,
        figsize=(15, 4.2 * num_rows),
        gridspec_kw={"width_ratios": [2.2, 1]},
        squeeze=False,
    )
    fig.patch.set_facecolor(background_color)

    feature_title = (
        MACRO_TITLES.get(target_feature, target_feature)
        if target_type == "asset-macro"
        else target_feature
    )

    fig.suptitle(
        f"Análise de Defasagem Temporal: {target_asset} vs {feature_title} "
        f"(Janela: {window_size}M · {y_label})",
        fontsize=16,
        y=1.0,
    )

    base_df = calculate_rolling_correlations(
        master_data,
        asset_list,
        window_size=window_size,
        method=method,
        target=target_type,
    )
    base_series = base_df.xs(target_asset, level=1)[target_feature]

    colors = styles.palette("lags", num_rows)
    base_color = styles.COLORS["neutral"]

    for i in range(num_rows):
        lag = active_lags[i]
        ax_line, ax_stats = axes[i, 0], axes[i, 1]
        ax_line.set_facecolor(background_color)
        ax_stats.set_facecolor(background_color)

        lag_df = calculate_rolling_correlations(
            master_data,
            asset_list,
            window_size=window_size,
            method=method,
            target=target_type,
            lag=lag,
            lag_features=[target_feature],
        )
        lag_series = lag_df.xs(target_asset, level=1)[target_feature]

        # Estatísticas sobre a sobreposição comum (delta justo, mesmo suporte).
        aligned = pd.concat({"base": base_series, "lag": lag_series}, axis=1).dropna()

        # ---- Esquerda: séries no tempo --------------------------------------
        if not aligned.empty:
            ax_line.fill_between(
                aligned.index,
                aligned["base"],
                aligned["lag"],
                color=colors[i],
                alpha=0.15,
                linewidth=0,
            )

        ax_line.plot(
            base_series.index,
            base_series,
            color=base_color,
            linestyle="--",
            linewidth=styles.lw("secondary"),
            label="Original (Lag 0)",
            alpha=0.7,
        )
        ax_line.plot(
            lag_series.index,
            lag_series,
            color=colors[i],
            linewidth=styles.lw("primary"),
            label=f"Defasado (Lag {lag}M)",
        )

        if not is_dcor:
            ax_line.axhline(
                0,
                color=styles.COLORS["zero_line"],
                linestyle="--",
                linewidth=styles.lw("reference"),
            )

        ax_line.set_title(f"Lag {lag}M", fontsize=13)
        ax_line.set_ylabel(y_label)
        ax_line.set_ylim(*ylim)
        ax_line.set_xlabel("Tempo", fontsize=11)
        ax_line.legend(loc="upper right", fontsize=9)

        # ---- Direita: média ± desvio + delta --------------------------------
        if aligned.empty:
            ax_stats.text(
                0.5,
                0.5,
                "Sem sobreposição",
                ha="center",
                va="center",
                fontsize=11,
                color="#888888",
            )
            ax_stats.set_axis_off()
            continue

        base_mean, base_std = aligned["base"].mean(), aligned["base"].std()
        lag_mean, lag_std = aligned["lag"].mean(), aligned["lag"].std()
        d_mean, d_std = lag_mean - base_mean, lag_std - base_std

        means = [base_mean, lag_mean]
        stds = [base_std, lag_std]
        x = [0, 1]
        bars = ax_stats.bar(
            x,
            means,
            yerr=stds,
            width=0.6,
            color=[base_color, colors[i]],
            alpha=0.85,
            capsize=6,
            error_kw=dict(ecolor="#444444", lw=1.4),
        )

        for xi, m, s in zip(x, means, stds):
            ax_stats.annotate(
                f"μ={m:.2f}\nσ={s:.2f}",
                xy=(xi, m),
                xytext=(0, 8 if m >= 0 else -22),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#333333",
            )

        if not is_dcor:
            ax_stats.axhline(
                0,
                color=styles.COLORS["zero_line"],
                linestyle="--",
                linewidth=styles.lw("reference"),
            )

        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(["Lag 0", f"Lag {lag}M"], fontsize=10)
        ax_stats.set_ylim(*ylim)
        ax_stats.set_ylabel(y_label, fontsize=10)
        ax_stats.set_title(f"Δμ {d_mean:+.2f}   ·   Δσ {d_std:+.2f}", fontsize=12)

    plt.tight_layout()
    plt.show()


def _macro_signature_snapshot(
    master_data: pd.DataFrame,
    asset_list: list = None,
    months_ago: int = None,
    macro_features: list = None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    top_n: int = None,
) -> tuple:
    """
    Monta o snapshot (asset x IMec) da macro_signature em uma única data.

    Helper compartilhado pelas funções `plot_asset_macro_signature_*`: resolve a
    seleção de ativos, calcula a rolling correlation asset↔IMec, resolve
    `months_ago` em uma data e devolve o recorte daquela data com as colunas
    encurtadas via `MACRO_TITLES`.

    Parameters
    ----------
    top_n : int or None
        Quando `asset_list is None`, limita aos `top_n` ativos com mais
        observações. `None` → todo o universo (mesmo conjunto que a
        clusterização vê). Ignorado se `asset_list` for passado.

    Returns
    -------
    (display, target_date) : (pd.DataFrame, pd.Timestamp)
        `display` é o snapshot asset x IMec (colunas encurtadas, linhas na ordem
        de `selected_assets`); `target_date` é a data efetivamente usada.
    """
    if macro_features is None:
        macro_features = MACRO_FEATURES

    universe = [
        c
        for c in master_data.columns
        if c not in MACRO_FEATURES and c != "MARKET_INDEX"
    ]

    if asset_list is None:
        counts = master_data[universe].count().sort_values(ascending=False)
        selected_assets = (
            counts.index.tolist()
            if top_n is None
            else counts.head(top_n).index.tolist()
        )
    else:
        selected_assets = [a for a in asset_list if a in universe]
        if not selected_assets:
            raise ValueError("Nenhum ativo válido em asset_list.")

    rolling_corr_df = calculate_rolling_correlations(
        master_data,
        selected_assets,
        window_size=window_size,
        method=method,
        target="asset-macro",
        lag=lag,
        lag_features=macro_features,
    )

    available_dates = rolling_corr_df.index.get_level_values(0).unique().sort_values()
    last_date = available_dates[-1]

    if months_ago is None:
        target_date = last_date
    else:
        approx = last_date - pd.DateOffset(months=months_ago)
        candidates = available_dates[available_dates <= approx]
        if len(candidates) == 0:
            raise ValueError(
                f"months_ago={months_ago} excede o histórico disponível "
                f"(primeira data: {available_dates[0].strftime('%Y-%m')})."
            )
        target_date = candidates[-1]

    snapshot = rolling_corr_df.xs(target_date, level=0)[macro_features]
    snapshot = snapshot.reindex(selected_assets)

    display = snapshot.copy()
    display.columns = [
        MACRO_TITLES.get(c, c).split("(")[0].strip() for c in display.columns
    ]
    return display, target_date


def plot_asset_macro_signature_heatmap(
    master_data: pd.DataFrame,
    asset_list: list = None,
    months_ago: int = None,
    macro_features: list = None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Heatmap (asset x IMEC) da macro_signature em uma única data.

    Parameters
    ----------
    asset_list : list or None
        Ativos a plotar. None → top 10 com mais observações em master_data
        (cap de legibilidade da heatmap).
    months_ago : int or None
        Quantos meses recuar a partir da última data disponível. None → última.
    """
    display, target_date = _macro_signature_snapshot(
        master_data,
        asset_list=asset_list,
        months_ago=months_ago,
        macro_features=macro_features,
        window_size=window_size,
        method=method,
        lag=lag,
        top_n=10,
    )
    selected_assets = display.index.tolist()

    is_dcor = method == "dcor"
    vmin, vmax = (0.0, 1.0) if is_dcor else (-1.0, 1.0)
    center = None if is_dcor else 0.0
    cmap = styles.heatmap_cmap(method)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(selected_assets) + 2)))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    sns.heatmap(
        display,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": f"Correlação ({method})"},
    )

    lag_suffix = f"  |  Lag: {lag}M" if lag > 0 else ""
    ax.set_title(
        f"Macro signature  |  {target_date.strftime('%Y-%m')}  |  "
        f"Janela: {window_size}M{lag_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Ativo")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def plot_asset_macro_signature_distribution(
    master_data: pd.DataFrame,
    asset_list: list = None,
    months_ago: int = None,
    macro_features: list = None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    top_n: int = None,
    standardize: bool = True,
    show_points: bool = True,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Boxplots da distribuição (entre ativos) dos valores da macro_signature,
    um box por IMec, em uma única data.

    Diagnóstico complementar a `plot_asset_macro_signature_heatmap`: o heatmap
    mostra os valores célula a célula, mas para entender por que a clusterização
    separa mal o que importa é a *dispersão* de cada coluna. Se a distribuição de
    um IMec for estreita/concentrada, os ativos têm signatures parecidas naquela
    coordenada e não há estrutura para o K-Means separar — o que explica
    silhouettes baixos. Aceita os mesmos parâmetros do heatmap para que ambos
    possam ser plotados sobre o mesmo snapshot.

    Parameters
    ----------
    asset_list : list or None
        Ativos a considerar. None → todo o universo (mesmo conjunto que a
        clusterização vê), exceto se `top_n` for passado.
    months_ago : int or None
        Quantos meses recuar a partir da última data disponível. None → última.
    top_n : int or None
        Quando `asset_list is None`, limita aos `top_n` ativos com mais
        observações. None → todos. Ignorado se `asset_list` for passado.
    standardize : bool
        Se True, z-scora cada IMec (μ=0, σ=1, ddof=0) antes de plotar —
        reproduz o que o K-Means de fato enxerga, já que a clusterização aplica
        `StandardScaler` por janela. Use para comparar dispersão *entre* IMecs
        de forma justa (no espaço cru o StandardScaler já equaliza as variâncias).
    show_points : bool
        Sobrepõe os ativos individuais (stripplot) sobre cada box.
    """
    display, target_date = _macro_signature_snapshot(
        master_data,
        asset_list=asset_list,
        months_ago=months_ago,
        macro_features=macro_features,
        window_size=window_size,
        method=method,
        lag=lag,
        top_n=top_n,
    )
    if standardize:
        display = (display - display.mean()) / display.std(ddof=0)
    long = display.melt(var_name="IMec", value_name="corr").dropna(subset=["corr"])

    is_dcor = method == "dcor"
    if standardize:
        ylim = None
    else:
        ylim = (-0.05, 1.05) if is_dcor else (-1.1, 1.1)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(display.columns) + 2), 6))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    order = display.columns.tolist()
    sns.boxplot(
        data=long,
        x="IMec",
        y="corr",
        order=order,
        palette=styles.palette("distribution", len(order)),
        showfliers=not show_points,
        width=0.6,
        ax=ax,
    )
    if show_points:
        sns.stripplot(
            data=long,
            x="IMec",
            y="corr",
            order=order,
            color=styles.COLORS["stripplot"],
            size=4,
            alpha=0.55,
            jitter=0.2,
            ax=ax,
        )

    if standardize or not is_dcor:
        ax.axhline(
            0,
            color=styles.COLORS["zero_line"],
            linestyle="--",
            linewidth=styles.lw("reference"),
            alpha=0.7,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)

    n_assets = long.groupby("IMec")["corr"].count().reindex(order)
    ax.set_xticklabels(
        [f"{lbl}\n(n={int(n)})" for lbl, n in zip(order, n_assets)],
        fontsize=10,
    )

    lag_suffix = f"  |  Lag: {lag}M" if lag > 0 else ""
    std_suffix = "  |  z-score por IMec" if standardize else ""
    ax.set_title(
        f"Distribuição da macro signature  |  {target_date.strftime('%Y-%m')}  |  "
        f"Janela: {window_size}M{lag_suffix}{std_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Correlação z-scorada (σ por IMec)" if standardize else f"Correlação ({method})"
    )
    ax.set_xlabel("")
    plt.tight_layout()
    plt.show()


def plot_asset_macro_signature_histograms(
    master_data: pd.DataFrame,
    asset_list: list = None,
    months_ago: int = None,
    macro_features: list = None,
    window_size: int = 6,
    method: str = "spearman",
    lag: int = 0,
    top_n: int = None,
    standardize: bool = True,
    bins: int = 20,
    kde: bool = True,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Histogramas (com KDE opcional) da macro_signature, um painel por IMec em
    grid 2x2, em uma única data.

    Variante de `plot_asset_macro_signature_distribution`: o boxplot resume a
    dispersão em cinco números e esconde a *forma* da distribuição, enquanto o
    histograma/KDE revela multimodalidade, assimetria e concentração. Para o
    diagnóstico de clusterização ruim isso é o que importa — uma distribuição
    unimodal e concentrada em um IMec significa que aquela coordenada não
    contribui para separar grupos; já bimodalidade sugere estrutura latente que o
    K-Means deveria capturar. Aceita os mesmos parâmetros do heatmap/boxplot para
    plotar sobre o mesmo snapshot.

    Parameters
    ----------
    asset_list : list or None
        Ativos a considerar. None → todo o universo (mesmo conjunto que a
        clusterização vê), exceto se `top_n` for passado.
    top_n : int or None
        Quando `asset_list is None`, limita aos `top_n` ativos com mais
        observações. None → todos. Ignorado se `asset_list` for passado.
    standardize : bool
        Se True, z-scora cada IMec (μ=0, σ=1, ddof=0) antes de plotar —
        reproduz o espaço que o K-Means enxerga (`StandardScaler` por janela). A
        *forma* da distribuição é invariante ao z-score, então a leitura de
        multimodalidade não muda; serve para comparar IMecs na mesma escala.
    bins : int
        Número de bins do histograma.
    kde : bool
        Sobrepõe estimativa de densidade (KDE) ao histograma.
    """
    display, target_date = _macro_signature_snapshot(
        master_data,
        asset_list=asset_list,
        months_ago=months_ago,
        macro_features=macro_features,
        window_size=window_size,
        method=method,
        lag=lag,
        top_n=top_n,
    )
    if standardize:
        display = (display - display.mean()) / display.std(ddof=0)

    is_dcor = method == "dcor"
    if standardize:
        xlim = None
    else:
        xlim = (-0.05, 1.05) if is_dcor else (-1.1, 1.1)
    order = display.columns.tolist()
    colors = styles.palette("distribution", len(order))

    sns.set_theme(style="whitegrid")
    n_cols = 2
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4), squeeze=False
    )
    fig.patch.set_facecolor(background_color)
    axes_flat = axes.flatten()

    for ax, feat, color in zip(axes_flat, order, colors):
        ax.set_facecolor(background_color)
        values = display[feat].dropna()
        sns.histplot(
            values,
            bins=bins,
            kde=kde,
            color=color,
            edgecolor="white",
            alpha=0.75,
            ax=ax,
        )
        if standardize:
            # μ=0/σ=1 por construção; a linha em 0 é a média.
            ax.axvline(
                0,
                color=styles.COLORS["zero_line"],
                linestyle="--",
                linewidth=styles.lw("reference"),
                label="μ=0 (z-score)",
            )
        else:
            mean_val = values.mean()
            ax.axvline(
                mean_val,
                color=styles.COLORS["zero_line"],
                linestyle="--",
                linewidth=styles.lw("reference"),
                label=f"μ={mean_val:.2f}  σ={values.std():.2f}",
            )
            if not is_dcor:
                ax.axvline(
                    0,
                    color=styles.COLORS["ref_line"],
                    linestyle=":",
                    linewidth=styles.lw("reference"),
                    alpha=0.7,
                )
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(f"{feat}  (n={len(values)})", fontsize=11)
        ax.set_xlabel(
            "z-score" if standardize else f"Correlação ({method})", fontsize=9
        )
        ax.set_ylabel("Nº de ativos", fontsize=9)
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    for ax in axes_flat[len(order) :]:
        fig.delaxes(ax)

    lag_suffix = f"  |  Lag: {lag}M" if lag > 0 else ""
    std_suffix = "  |  z-score por IMec" if standardize else ""
    fig.suptitle(
        f"Distribuição da macro signature  |  {target_date.strftime('%Y-%m')}  |  "
        f"Janela: {window_size}M{lag_suffix}{std_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# Camada viz/ — séries brutas (complemento das curvas de correlação)
#
# As funções acima plotam a EVOLUÇÃO da correlação móvel. Estas plotam as séries
# BRUTAS por trás dela, para que o co-movimento que origina a correlação fique
# visível: o retorno do ativo vs a série do IMec (ativo↔IMec) e o índice de preço
# reconstruído (ativo↔ativo) — a MESMA reconstrução usada na clusterização DTW
# "raw". Nenhuma correlação é recalculada aqui; é só leitura do master_df. Não se
# aplica lag (são as séries observadas como são).
# =============================================================================


def _reconstruct_price_index(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Reconstrói um índice de preço (base `base`) a partir da série de retornos
    mensais em % via produto acumulado — a MESMA reconstrução da clusterização DTW
    "raw" (`clustering._build_series_dataset`, series_mode="raw"). O master_df
    guarda apenas retornos; a escala absoluta do preço não é recuperável, então o
    índice começa em `base` no primeiro retorno válido do ativo.
    """
    ret = returns.dropna()
    return base * (1.0 + ret / 100.0).cumprod()


def plot_asset_macro_raw_series(
    master_data: pd.DataFrame,
    target_asset: str,
    macro_features: list,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Séries brutas por trás da correlação móvel ativo↔IMec.

    Um painel por IMec: o retorno mensal do ativo (eixo esquerdo) sobreposto à
    série do IMec (eixo direito, escala própria) no mesmo eixo de tempo. Cada série
    mantém sua unidade real (twin axis) para deixar o co-movimento bruto visível
    sem distorcer magnitudes. Só lê o master_df; não recalcula correlações nem
    aplica lag.
    """
    feats = [f for f in macro_features if f in master_data.columns]
    if not feats:
        raise ValueError("Nenhuma macro_feature válida em macro_features.")

    n = len(feats)
    n_cols = 1 if n == 1 else 2
    n_rows = int(np.ceil(n / n_cols))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 7.5, n_rows * 4.0), squeeze=False
    )
    fig.patch.set_facecolor(background_color)
    axes_flat = axes.flatten()

    asset_color = styles.COLORS["asset_emphasis"]
    macro_colors = [styles.macro_color(f) for f in feats]
    asset_series = master_data[target_asset].dropna()

    for ax_ret, feat, m_color in zip(axes_flat, feats, macro_colors):
        ax_ret.set_facecolor(background_color)
        line_ret = ax_ret.plot(
            asset_series.index,
            asset_series,
            color=asset_color,
            linewidth=styles.lw("primary"),
            label=f"{target_asset} (retorno)",
        )[0]
        ax_ret.axhline(
            0,
            color=styles.COLORS["zero_line"],
            linestyle="--",
            linewidth=styles.lw("reference"),
            alpha=0.5,
        )
        ax_ret.set_ylabel(f"Retorno {target_asset} (%)", color=asset_color, fontsize=10)
        ax_ret.tick_params(axis="y", labelcolor=asset_color)

        ax_macro = ax_ret.twinx()
        macro_series = master_data[feat].dropna()
        short = _MACRO_SHORT_LABELS.get(feat, feat)
        line_macro = ax_macro.plot(
            macro_series.index,
            macro_series,
            color=m_color,
            linewidth=styles.lw("primary"),
            label=short,
        )[0]
        ax_macro.set_ylabel(MACRO_TITLES.get(feat, feat), color=m_color, fontsize=9)
        ax_macro.tick_params(axis="y", labelcolor=m_color)
        ax_macro.grid(False)

        ax_ret.set_title(MACRO_TITLES.get(feat, feat), fontsize=11)
        ax_ret.legend(
            handles=[line_ret, line_macro], loc="upper left", fontsize=8, framealpha=0.9
        )

    for ax in axes_flat[n:]:
        fig.delaxes(ax)

    fig.suptitle(
        f"Séries brutas — {target_asset} (retorno) vs IMecs",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_assets_raw_price(
    master_data: pd.DataFrame,
    target_asset: str,
    other_assets,
    base: float = 100.0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Índice de preço reconstruído (base `base`) do alvo e dos ativos comparados.

    Séries brutas por trás da correlação móvel ativo↔ativo: a MESMA reconstrução da
    clusterização DTW "raw" (cumprod dos retornos). Cada ativo é rebaseado em
    `base` no seu primeiro retorno válido (painel desbalanceado), em linha com o
    que o DTW enxerga. Só lê o master_df; não aplica lag.
    """
    others = (
        list(other_assets)
        if isinstance(other_assets, (list, tuple))
        else [other_assets]
    )
    others = [
        a
        for a in others
        if a != target_asset and a in master_data.columns and a != "MARKET_INDEX"
    ]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    if target_asset in master_data.columns:
        price = _reconstruct_price_index(master_data[target_asset], base=base)
        if not price.empty:
            ax.plot(
                price.index,
                price,
                color=styles.COLORS["asset_emphasis"],
                linewidth=styles.lw("emphasis"),
                label=f"{target_asset} (alvo)",
                zorder=5,
            )

    colors = styles.palette("assets", max(len(others), 1))
    for color, asset in zip(colors, others):
        price = _reconstruct_price_index(master_data[asset], base=base)
        if price.empty:
            continue
        ax.plot(
            price.index,
            price,
            color=color,
            linewidth=styles.lw("primary"),
            alpha=0.9,
            label=asset,
        )

    ax.axhline(
        base,
        color=styles.COLORS["ref_line"],
        linestyle=":",
        linewidth=styles.lw("reference"),
        alpha=0.7,
    )
    ax.set_title(
        f"Séries brutas — índice de preço reconstruído (base {base:.0f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Tempo", fontsize=12)
    ax.set_ylabel(f"Índice de preço (base {base:.0f})", fontsize=12)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Ativos",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.show()
