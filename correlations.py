import os
import pickle

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from util_funcs import (
    load_normalize_ibc_data,
    load_normalize_ipea_data,
    treat_candle_data,
)

MACRO_FEATURES = [
    "macro_selic_change",
    "macro_ipca",
    "macro_dollar_var",
    "macro_ibcbr_var",
]

MACRO_TITLES = {
    "macro_selic_change": "Variação da Selic (%)",
    "macro_ipca": "Nível do IPCA (pontos percentuais)",
    "macro_dollar_var": "Variação do Dólar (%)",
    "macro_ibcbr_var": "Variação do IBC-Br (%)",
}

CANDLE_VARIABLES = ["OPEN", "HIGH", "LOW", "CLOSE"]


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


def calculate_static_correlations(
    master_data: pd.DataFrame,
    asset_names: list,
    method: str = "spearman",
    force: bool = False,
) -> pd.DataFrame:
    """
    Calcula a correlação estática entre os ativos e os IMecs para todo o período.
    Implementa cache local automático via Parquet diferenciado pelo método escolhido.
    """
    available_assets = [c for c in asset_names if c in master_data.columns]

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"static_corr_{method}.parquet")

    if os.path.exists(cache_path) and not force:
        cached_df = pd.read_parquet(cache_path)
        valid_idx = [a for a in available_assets if a in cached_df.index]
        return cached_df.loc[valid_idx, MACRO_FEATURES]

    if method == "dcor":
        dcor_frame = pd.DataFrame(
            index=available_assets, columns=MACRO_FEATURES, dtype=float
        )
        for asset in available_assets:
            for macro in MACRO_FEATURES:
                valid_pairs = master_data[[asset, macro]].dropna()
                if len(valid_pairs) > 1:
                    dcor_frame.loc[asset, macro] = dcor.distance_correlation(
                        valid_pairs[asset].values, valid_pairs[macro].values
                    )
                else:
                    dcor_frame.loc[asset, macro] = np.nan
        result_df = dcor_frame
    else:
        corr_matrix = master_data.corr(method=method)
        result_df = corr_matrix.loc[available_assets, MACRO_FEATURES]

    result_df.to_parquet(cache_path)

    return result_df


def calculate_rolling_correlations(
    master_data: pd.DataFrame,
    asset_list: list,
    window_size: int,
    method: str = "spearman",
    target: str = "asset-macro",
    force: bool = False,
) -> pd.DataFrame:
    """
    Calcula matrizes de correlação móveis ignorando o MARKET_INDEX.
    Implementa cache local otimizado via Pickle comprimido para MultiIndex 3D,
    diferenciando por janela, método e alvo da análise.
    """
    clean_list = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"rolling_corr_W{window_size}_{method}_{target}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path) and not force:
        with open(cache_path, "rb") as f:
            cached_result = pickle.load(f)

        idx = pd.IndexSlice
        target_cols = clean_list if target == "asset-asset" else MACRO_FEATURES
        valid_assets = [a for a in clean_list if a in cached_result.index.levels[1]]

        print("Loaded rolling correlations from cache.")
        return cached_result.loc[idx[:, valid_assets], target_cols]

    print("Calculating rolling correlations...")
    if method == "pearson":
        rolling_corr = master_data.rolling(window=window_size).corr()
    else:
        frames = []
        keys = []
        min_periods = max(4, int(window_size * 0.5))

        for i in range(window_size - 1, len(master_data)):
            window = master_data.iloc[i - window_size + 1 : i + 1]

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

            keys.append(master_data.index[i])

        rolling_corr = pd.concat(frames, keys=keys, names=["TIMESTAMP", "Variável"])

    idx = pd.IndexSlice

    if target == "asset-macro":
        result = rolling_corr.loc[idx[:, clean_list], MACRO_FEATURES]
    elif target == "asset-asset":
        result = rolling_corr.loc[idx[:, clean_list], clean_list]
    else:
        raise ValueError("Target parameter must be 'asset-macro' or 'asset-asset'")

    final_result = result.ffill().dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(final_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return final_result


def _extract_historical_features(
    master_data: pd.DataFrame,
    asset_list: list,
    macro_features: list,
    window_size: int,
    evaluation_period: int,
    lag: int,
    target_date: str = None,
) -> tuple:
    """Extrai features históricas garantindo a exclusão de benchmarks sintéticos."""
    clean_assets = [a for a in asset_list if a != "MARKET_INDEX"]

    working_data = master_data.copy()

    if lag > 0:
        working_data[macro_features] = working_data[macro_features].shift(lag)

    rolling_corr_df = calculate_rolling_correlations(
        working_data, clean_assets, window_size=window_size, target="asset-macro"
    )

    available_dates = rolling_corr_df.index.get_level_values(0).unique()

    if target_date is None:
        selected_date = available_dates[-1]
    else:
        selected_date = pd.to_datetime(target_date)
        if selected_date not in available_dates:
            selected_date = available_dates[available_dates <= selected_date][-1]

    history_df = rolling_corr_df[
        rolling_corr_df.index.get_level_values(0) <= selected_date
    ]
    dates_in_period = history_df.index.get_level_values(0).unique()[-evaluation_period:]
    period_data = history_df.loc[
        history_df.index.get_level_values(0).isin(dates_in_period)
    ]

    mean_df = period_data.groupby(level=1)[macro_features].mean()
    std_df = period_data.groupby(level=1)[macro_features].std()

    mean_df.columns = [f"{c}_mean" for c in mean_df.columns]
    std_df.columns = [f"{c}_std" for c in std_df.columns]

    features_df = pd.concat([mean_df, std_df], axis=1).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)

    return features_df, scaled_data, scaler, selected_date


def calculate_macro_kmeans_clusters(
    master_data: pd.DataFrame,
    asset_list: list,
    macro_features: list,
    n_clusters: int = 4,
    window_size: int = 12,
    evaluation_period: int = 24,
    lag: int = 0,
    target_date: str = None,
    apply_pca: bool = False,
) -> tuple:
    """Realiza a clusterização K-Means ignorando o MARKET_INDEX."""
    features_df, scaled_data, scaler, selected_date = _extract_historical_features(
        master_data,
        asset_list,
        macro_features,
        window_size,
        evaluation_period,
        lag,
        target_date,
    )

    if apply_pca:
        pca_model = PCA(n_components=0.90, random_state=42)
        clustering_data = pca_model.fit_transform(scaled_data)
    else:
        clustering_data = scaled_data

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data)
    sil_score = (
        silhouette_score(clustering_data, cluster_labels)
        if len(set(cluster_labels)) > 1
        else -1.0
    )

    clustered_data = features_df.copy()
    clustered_data["CLUSTER"] = cluster_labels

    if apply_pca:
        orig_scaled = pca_model.inverse_transform(kmeans.cluster_centers_)
        centroids_vals = scaler.inverse_transform(orig_scaled)
    else:
        centroids_vals = scaler.inverse_transform(kmeans.cluster_centers_)

    centroids = pd.DataFrame(centroids_vals, columns=features_df.columns)
    return clustered_data, centroids, selected_date, sil_score


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

    if not is_single_feature:
        colors = sns.color_palette("husl", len(features_to_plot))
    else:
        window_colors = sns.color_palette("viridis", num_windows)

    working_data = master_data.copy()
    if lag > 0:
        working_data[features_to_plot] = working_data[features_to_plot].shift(lag)

    calculated_series = []

    for ax_idx, window in enumerate(window_sizes):
        rolling_corr_df = calculate_rolling_correlations(
            working_data,
            asset_list,
            window_size=window,
            target=target_type,
            method=method,
        )
        calculated_series.append(rolling_corr_df)

    all_dates = pd.concat(
        [df.index.get_level_values(0).to_series() for df in calculated_series]
    )
    shared_min_date = all_dates.min()
    shared_max_date = all_dates.max()

    for ax_idx, (ax, window) in enumerate(zip(axes, window_sizes)):
        ax.set_facecolor(background_color)
        rolling_corr_df = calculated_series[ax_idx]

        for i, feature in enumerate(features_to_plot):
            time_series = rolling_corr_df.xs(target_asset, level=1)[feature]

            line_color = window_colors[ax_idx] if is_single_feature else colors[i]
            label = (
                MACRO_TITLES.get(feature, feature)
                if target_type == "asset-macro"
                else feature
            )

            ax.plot(
                time_series.index,
                time_series,
                color=line_color,
                linewidth=2.0,
                label=label,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

        ax.set_xlim(shared_min_date, shared_max_date)

        if is_single_feature:
            volatility = time_series.std()
            if method == "dcor":
                ax.set_title(f"Janela de {window} meses", fontsize=13)
            else:
                sign_flips = ((time_series * time_series.shift(1)) < 0).sum()
                ax.set_title(
                    f"Janela de {window} meses (Flips: {sign_flips})",
                    fontsize=13,
                )
        else:
            ax.set_title(f"Janela de {window} meses", fontsize=13)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                title=legend_title,
                borderaxespad=0.0,
            )

        if method == "dcor":
            ax.set_ylabel("Distance Corr.")
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel(method.capitalize())
            ax.set_ylim(-1.1, 1.1)

    plt.xlabel("Tempo", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_rolling_windows_comparison(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    target_macro: str,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = "#ffffff",
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
    background_color: str = "#ffffff",
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
    background_color: str = "#ffffff",
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


def plot_top_n_correlated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = "#ffffff",
    lag: int = 0,
    method: str = "spearman",
    abs_corr: bool = False,
) -> None:
    """Plota a evolução da correlação móvel dos ativos com maior dependência global com o alvo."""
    clean_assets = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]
    working_data = master_data[clean_assets].copy()

    if lag > 0:
        cols_to_shift = [c for c in clean_assets if c != target_asset]
        working_data[cols_to_shift] = working_data[cols_to_shift].shift(lag)

    target_corrs = pd.Series(
        index=[a for a in clean_assets if a != target_asset], dtype=float
    )

    if method == "dcor":
        for a in target_corrs.index:
            valid_pairs = working_data[[target_asset, a]].dropna()
            if len(valid_pairs) > 3:
                target_corrs[a] = dcor.distance_correlation(
                    valid_pairs[target_asset].values, valid_pairs[a].values
                )
        top_assets = target_corrs.nlargest(top_n).index.tolist()
    else:
        corr_matrix = working_data.corr(method=method)
        target_corrs = corr_matrix[target_asset].drop(target_asset)

        top_assets = (
            target_corrs.nlargest(top_n).index.tolist()
            if not abs_corr
            else target_corrs.abs().nlargest(top_n).index.tolist()
        )

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=top_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução dos Top {top_n} Correlacionados com {target_asset}",
        legend_title="Ativos",
        lag=lag,
        method=method,
    )


def plot_top_n_inversely_correlated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = "#ffffff",
    lag: int = 0,
) -> None:
    """Plota a evolução da correlação móvel dos ativos menos correlacionados com o ativo alvo com suporte a lag."""
    available_assets = [c for c in asset_list if c in master_data.columns]

    working_data = master_data[available_assets].copy()
    if lag > 0:
        cols_to_shift = [c for c in available_assets if c != target_asset]
        working_data[cols_to_shift] = working_data[cols_to_shift].shift(lag)

    corr_matrix = working_data.corr(method="spearman")
    least_assets = (
        corr_matrix[target_asset].drop(target_asset).nsmallest(top_n).index.tolist()
    )

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=least_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução dos Top {top_n} Menos Correlacionados com {target_asset}",
        legend_title="Ativos",
        lag=lag,
    )


def plot_top_n_uncorrelated_assets(
    master_data: pd.DataFrame,
    asset_list: list,
    target_asset: str,
    top_n: int = 3,
    window_sizes: list = [3, 6, 12, 24],
    background_color: str = "#ffffff",
    lag: int = 0,
    method: str = "spearman",
) -> None:
    """Plota a evolução dos ativos mais estatisticamente independentes (próximos a zero) com o alvo."""
    clean_assets = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]
    working_data = master_data[clean_assets].copy()

    if lag > 0:
        cols_to_shift = [c for c in clean_assets if c != target_asset]
        working_data[cols_to_shift] = working_data[cols_to_shift].shift(lag)

    target_corrs = pd.Series(
        index=[a for a in clean_assets if a != target_asset], dtype=float
    )

    if method == "dcor":
        for a in target_corrs.index:
            valid_pairs = working_data[[target_asset, a]].dropna()
            if len(valid_pairs) > 3:
                target_corrs[a] = dcor.distance_correlation(
                    valid_pairs[target_asset].values, valid_pairs[a].values
                )

        top_assets = target_corrs.nsmallest(top_n).index.tolist()
    else:
        corr_matrix = working_data.corr(method=method)
        target_corrs = corr_matrix[target_asset].drop(target_asset)

        top_assets = target_corrs.abs().nsmallest(top_n).index.tolist()

    _plot_rolling_subplots(
        master_data=master_data,
        asset_list=asset_list,
        target_asset=target_asset,
        features_to_plot=top_assets,
        target_type="asset-asset",
        window_sizes=window_sizes,
        background_color=background_color,
        title=f"Evolução dos Top {top_n} Ativos Neutros/Descorrelacionados com {target_asset}",
        legend_title="Ativos",
        lag=lag,
        method=method,
    )


def plot_macro_kmeans_clusters(
    clustered_data: pd.DataFrame,
    centroids: pd.DataFrame,
    selected_date: pd.Timestamp,
    sil_score: float,
    window_size: int,
    evaluation_period: int = 24,
    lag: int = 0,
    apply_pca: bool = False,
    background_color: str = "#ffffff",
) -> None:
    """
    Gera a projeção PCA e o Heatmap dos centróides, destacando a qualidade
    do agrupamento através do Silhouette Score no título.
    """
    feature_cols = [c for c in clustered_data.columns if c != "CLUSTER"]
    n_clusters = len(centroids)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(background_color)
    axes[0].set_facecolor(background_color)
    axes[1].set_facecolor(background_color)

    lag_suffix = f" | Lag: {lag}M" if lag > 0 else ""
    pca_suffix = " | Pré-PCA: ON" if apply_pca else ""

    fig.suptitle(
        f"K-Means Histórico (Ref: {selected_date.strftime('%Y-%m')}){lag_suffix}{pca_suffix} \n"
        f"Silhouette Score: {sil_score:.3f} (Janela: {window_size}M, Histórico: {evaluation_period}M)",
        fontsize=16,
        y=1.08,
        fontweight="bold",
    )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustered_data[feature_cols])
    pca_viz = PCA(n_components=2)
    pca_components = pca_viz.fit_transform(scaled_data)

    plot_df = clustered_data.copy()
    plot_df["PCA1"] = pca_components[:, 0]
    plot_df["PCA2"] = pca_components[:, 1]

    colors = sns.color_palette("tab10", n_clusters)

    sns.scatterplot(
        data=plot_df,
        x="PCA1",
        y="PCA2",
        hue="CLUSTER",
        palette=colors,
        s=100,
        ax=axes[0],
        legend="full",
    )

    for idx, row in plot_df.iterrows():
        axes[0].annotate(
            idx,
            (row["PCA1"], row["PCA2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    axes[0].set_title(f"Projeção Visual 2D ({n_clusters} Clusters)", fontsize=13)
    axes[0].set_xlabel(
        f"Componente Principal 1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)"
    )
    axes[0].set_ylabel(
        f"Componente Principal 2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)"
    )

    display_centroids = centroids.copy()
    translated_cols = []
    for c in display_centroids.columns:
        if c.endswith("_mean"):
            base_col = c.replace("_mean", "")
            translated_cols.append(
                f"Média: {MACRO_TITLES.get(base_col, base_col).split('(')[0].strip()}"
            )
        elif c.endswith("_std"):
            base_col = c.replace("_std", "")
            translated_cols.append(
                f"Volat.: {MACRO_TITLES.get(base_col, base_col).split('(')[0].strip()}"
            )

    display_centroids.columns = translated_cols

    sns.heatmap(
        display_centroids,
        annot=True,
        cmap="vlag",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        ax=axes[1],
        cbar_kws={"label": "Estatística da Correlação"},
    )

    axes[1].set_title("Perfil Histórico dos Clusters (Centróides Reais)", fontsize=13)
    axes[1].set_ylabel("Cluster")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
