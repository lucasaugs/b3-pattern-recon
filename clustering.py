import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import styles
from correlations import (
    MACRO_TITLES,
    calculate_rolling_correlations,
    full_period_macro_signature,
)

# =============================================================================
# Camada clustering/ — clusterização por janela sobre a macro_signature
#
# Para cada timestamp da rolling correlation (asset x macro_features), rodamos
# uma clusterização independente. Os labels de cada janela são canonizados via
# matching húngaro (distância euclidiana) contra a média corrente dos centroides
# canonizados das janelas anteriores, garantindo que "cluster 0" represente o
# mesmo perfil macroeconômico ao longo do tempo. A atribuição final de cada
# ativo é a moda dos seus labels canonizados; o flip rate mede transições
# entre janelas consecutivas.
# =============================================================================


def _centroids_from_labels(
    scaled_data: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """
    Centroide = média dos pontos de cada cluster no espaço escalado.

    Para métodos sem centroide nativo (hierárquico, espectral). Cluster vazio
    (raro) recebe o vetor zero, que no espaço escalado pelo StandardScaler é o
    centroide global — fallback neutro que não quebra a canonização húngara.
    """
    n_features = scaled_data.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = scaled_data[mask].mean(axis=0)
    return centroids


def _fit_clustering(scaled_data: np.ndarray, n_clusters: int, method: str) -> tuple:
    """
    Ajusta o clusterizador e retorna (labels, centroides no espaço escalado).

    KMeans expõe centroides nativamente. Hierárquico e espectral não têm
    centroide explícito, então ele é derivado pela média dos pontos de cada
    cluster (`_centroids_from_labels`) — suficiente para a canonização entre
    janelas e para os plots de perfil/centroide a jusante.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)
        return labels, model.cluster_centers_
    if method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(scaled_data)
        return labels, _centroids_from_labels(scaled_data, labels, n_clusters)
    if method == "spectral":
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="rbf",
            assign_labels="kmeans",
            n_init=10,
        )
        labels = model.fit_predict(scaled_data)
        return labels, _centroids_from_labels(scaled_data, labels, n_clusters)
    raise ValueError(
        f"cluster_method '{method}' não suportado "
        "(use 'kmeans', 'hierarchical' ou 'spectral')."
    )


def _canonicalize_labels(
    new_centroids: np.ndarray, reference_centroids: np.ndarray
) -> np.ndarray:
    """Retorna mapping[raw_label] -> canonical_label via matching húngaro."""
    cost = np.linalg.norm(
        new_centroids[:, None, :] - reference_centroids[None, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = np.empty(len(new_centroids), dtype=int)
    mapping[row_ind] = col_ind
    return mapping


def cluster_assets_per_window(
    master_data: pd.DataFrame,
    asset_list: list,
    macro_features: list,
    n_clusters: int = 4,
    window_size: int = 6,
    lag: int = 0,
    ref_date: str = None,
    lookback_months: int = None,
    cluster_method: str = "kmeans",
    correlation_method: str = "spearman",
    exclude_features: list = None,
) -> dict:
    """
    Clusteriza ativos por janela sobre a macro_signature da rolling correlation.

    Parameters
    ----------
    window_size : int
        Janela (em meses) da rolling correlation que produz a macro_signature.
    lookback_months : int or None
        Horizonte de análise: nº de janelas mais recentes (contadas a partir de
        ref_date) sobre as quais rodar a clusterização. None = série inteira.
    ref_date : str or None
        Data de referência (extremo direito do horizonte). None = mais recente.
    cluster_method : str
        Backend de clusterização: 'kmeans', 'hierarchical' (Ward) ou 'spectral'
        (afinidade RBF). Todos rodam sobre a macro_signature z-scorada por janela.

    Returns
    -------
    dict com:
        assignments         : DataFrame (date x asset) de labels canonizados
        modal_label         : Series (asset -> cluster final por moda)
        centroids           : DataFrame (cluster x macro_features) — média dos
                              snapshots dos ativos atribuídos modalmente ao cluster
        centroids_per_window: dict {date -> DataFrame cluster x macro_features}
        flip_rate_per_asset : Series (asset -> fração de transições)
        flip_rate_mean      : float
        sil_per_window      : Series (date -> silhouette)
        sil_mean            : float
        ref_date            : Timestamp efetivamente usado
        window_dates        : DatetimeIndex das janelas clusterizadas
    """
    clean_assets = [a for a in asset_list if a != "MARKET_INDEX"]

    if exclude_features:
        macro_features = [f for f in macro_features if f not in exclude_features]
        if not macro_features:
            raise ValueError("exclude_features removeu todas as macro_features.")

    rolling_corr_df = calculate_rolling_correlations(
        master_data,
        clean_assets,
        window_size=window_size,
        method=correlation_method,
        target="asset-macro",
        lag=lag,
        lag_features=macro_features,
    )

    available_dates = rolling_corr_df.index.get_level_values(0).unique().sort_values()

    if len(available_dates) == 0:
        raise ValueError(
            f"Nenhuma janela com correlação válida (window_size={window_size}, "
            f"method={correlation_method}). Em dcor, janelas curtas (ex.: 3 meses) "
            "não atingem o mínimo de pares válidos e produzem um tensor vazio."
        )

    if ref_date is None:
        ref = available_dates[-1]
    else:
        ref = pd.to_datetime(ref_date)
        if ref not in available_dates:
            ref = available_dates[available_dates <= ref][-1]

    dates_up_to_ref = available_dates[available_dates <= ref]
    if lookback_months is None:
        window_dates = dates_up_to_ref
    else:
        window_dates = dates_up_to_ref[-lookback_months:]

    assignments: dict = {}
    centroids_per_window: dict = {}
    sil_per_window: dict = {}
    running_reference = None
    ref_count = 0

    for dt in window_dates:
        snapshot = rolling_corr_df.xs(dt, level=0)[macro_features].dropna()
        if len(snapshot) < n_clusters:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(snapshot.values)

        raw_labels, raw_centroids_scaled = _fit_clustering(
            X, n_clusters, cluster_method
        )
        centroids_orig = scaler.inverse_transform(raw_centroids_scaled)

        if running_reference is None:
            mapping = np.arange(n_clusters)
        else:
            mapping = _canonicalize_labels(centroids_orig, running_reference)

        canonical_labels = mapping[raw_labels]
        canonical_centroids = np.empty_like(centroids_orig)
        canonical_centroids[mapping] = centroids_orig

        if running_reference is None:
            running_reference = canonical_centroids.copy()
            ref_count = 1
        else:
            ref_count += 1
            running_reference = (
                running_reference
                + (canonical_centroids - running_reference) / ref_count
            )

        sil = (
            silhouette_score(X, canonical_labels)
            if len(set(canonical_labels)) > 1
            else np.nan
        )

        assignments[dt] = pd.Series(canonical_labels, index=snapshot.index)
        centroids_per_window[dt] = pd.DataFrame(
            canonical_centroids, columns=macro_features
        )
        sil_per_window[dt] = sil

    assignments_df = pd.DataFrame(assignments).T
    assignments_df.index.name = "date"
    sil_series = pd.Series(sil_per_window, name="silhouette")

    mode_df = assignments_df.mode(axis=0, dropna=True)
    modal_label = mode_df.iloc[0] if not mode_df.empty else pd.Series(dtype=float)
    modal_label.name = "cluster"

    def _asset_flips(col: pd.Series) -> float:
        s = col.dropna()
        if len(s) < 2:
            return np.nan
        return float((s.values[1:] != s.values[:-1]).mean())

    flip_rate = assignments_df.apply(_asset_flips, axis=0)
    flip_rate.name = "flip_rate"

    snapshot_all = rolling_corr_df.loc[
        rolling_corr_df.index.get_level_values(0).isin(window_dates)
    ][macro_features].copy()
    asset_level = snapshot_all.index.get_level_values(1)
    snapshot_all["_cluster"] = modal_label.reindex(asset_level).values
    final_centroids = (
        snapshot_all.dropna(subset=["_cluster"])
        .groupby("_cluster")[macro_features]
        .mean()
    )
    final_centroids.index = final_centroids.index.astype(int)
    final_centroids.index.name = "cluster"

    asset_signature = snapshot_all.groupby(level=1)[macro_features].mean()
    asset_signature.index.name = "asset"

    return {
        "assignments": assignments_df,
        "modal_label": modal_label.astype("Int64"),
        "centroids": final_centroids,
        "asset_signature": asset_signature,
        "centroids_per_window": centroids_per_window,
        "flip_rate_per_asset": flip_rate,
        "flip_rate_mean": float(flip_rate.mean()),
        "sil_per_window": sil_series,
        "sil_mean": float(sil_series.mean()),
        "ref_date": ref,
        "window_dates": window_dates,
        "correlation_method": correlation_method,
    }


def cluster_assets_full_period(
    master_data: pd.DataFrame,
    asset_list: list,
    macro_features: list,
    n_clusters: int = 4,
    lag: int = 0,
    cluster_method: str = "kmeans",
    correlation_method: str = "spearman",
    exclude_features: list = None,
) -> dict:
    """
    Clusteriza ativos sobre a macro_signature do período inteiro (correlação
    estática asset↔IMec), sem janelas e sem canonização.

    Retorna dict com a mesma forma de `cluster_assets_per_window` para preservar
    compatibilidade com `plot_clusters_3d` e `plot_clusters_pairs`. Campos que só
    fazem sentido na versão por janela (`centroids_per_window`, `flip_rate_*`,
    `sil_per_window`, `window_dates`, `assignments`) ficam vazios ou NaN.

    `cluster_method` aceita 'kmeans', 'hierarchical' (Ward) ou 'spectral' (RBF).
    """
    clean_assets = [a for a in asset_list if a != "MARKET_INDEX"]

    if exclude_features:
        macro_features = [f for f in macro_features if f not in exclude_features]
        if not macro_features:
            raise ValueError("exclude_features removeu todas as macro_features.")

    signature_df = full_period_macro_signature(
        master_data,
        clean_assets,
        method=correlation_method,
        lag=lag,
        lag_features=macro_features,
    )[macro_features].dropna()

    if len(signature_df) < n_clusters:
        raise ValueError(
            f"Apenas {len(signature_df)} ativos com signature válida; "
            f"insuficiente para {n_clusters} clusters."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(signature_df.values)

    raw_labels, raw_centroids_scaled = _fit_clustering(X, n_clusters, cluster_method)
    centroids_orig = scaler.inverse_transform(raw_centroids_scaled)

    sil = silhouette_score(X, raw_labels) if len(set(raw_labels)) > 1 else np.nan

    modal_label = pd.Series(
        raw_labels, index=signature_df.index, name="cluster"
    ).astype("Int64")

    centroids_df = pd.DataFrame(
        centroids_orig,
        columns=macro_features,
        index=pd.Index(range(n_clusters), name="cluster"),
    )

    asset_signature = signature_df.copy()
    asset_signature.index.name = "asset"

    ref = master_data.index.max()

    return {
        "assignments": pd.DataFrame(),
        "modal_label": modal_label,
        "centroids": centroids_df,
        "asset_signature": asset_signature,
        "centroids_per_window": {},
        "flip_rate_per_asset": pd.Series(dtype=float, name="flip_rate"),
        "flip_rate_mean": float("nan"),
        "sil_per_window": pd.Series(dtype=float, name="silhouette"),
        "sil_mean": float(sil) if not np.isnan(sil) else float("nan"),
        "ref_date": ref,
        "window_dates": pd.DatetimeIndex([]),
        "correlation_method": correlation_method,
    }


# =============================================================================
# Camada clustering/ — clusterização por DTW sobre as séries temporais
#
# Diferente das funções acima (que clusterizam vetores de assinatura macro), aqui
# clusterizamos a própria trajetória temporal de cada ativo com Dynamic Time
# Warping (tslearn.TimeSeriesKMeans). O DTW alinha séries deslocadas/esticadas no
# tempo, então agrupa ativos com *formato* de trajetória parecido mesmo que
# defasados. A qualidade é medida por um silhouette score calculado sobre a
# matriz de distâncias DTW (sklearn, metric="precomputed"), na MESMA escala
# [-1, 1] do silhouette das funções de assinatura — direto comparável.
# =============================================================================


def _resolve_temporal_scope(
    full_index: pd.DatetimeIndex,
    ref_date,
    lookback_months,
) -> tuple:
    """
    Resolve o recorte temporal (start, end) das funções de clusterização DTW.

    end  = `ref_date`, ou o último ponto de `full_index` quando `ref_date` é None.
    start = `end - lookback_months`, ou None (desde o início) quando
            `lookback_months` é None.

    O slicing associado é `start < t <= end` (start exclusivo), de modo que um
    `lookback_months=N` sobre dados mensais devolve exatamente N pontos. Sem
    `ref_date` nem `lookback_months` o escopo é o período inteiro — retorna
    (None, None) e nenhum corte é aplicado.
    """
    if ref_date is None and lookback_months is None:
        return None, None
    end = pd.Timestamp(ref_date) if ref_date is not None else full_index.max()
    start = (
        end - pd.DateOffset(months=lookback_months)
        if lookback_months is not None
        else None
    )
    return start, end


def _slice_temporal(series: pd.Series, start, end) -> pd.Series:
    """Recorta uma série indexada por data ao intervalo `start < t <= end`."""
    if end is not None:
        series = series[series.index <= end]
    if start is not None:
        series = series[series.index > start]
    return series


def _build_series_dataset(
    master_data: pd.DataFrame,
    asset_list: list,
    series_mode: str,
    min_periods: int,
    ref_date=None,
    lookback_months=None,
) -> tuple:
    """
    Monta o dataset de séries temporais por ativo para a clusterização DTW.

    series_mode:
        "returns" — usa a série de retornos mensais (colunas do master_df).
        "raw"     — reconstrói um índice de preço (base 100) a partir dos
                    retornos via produto acumulado `100 * cumprod(1 + r/100)`.
                    O master_df guarda apenas retornos, então "valores brutos"
                    aqui é a *trajetória de preço* reconstruída (a escala
                    absoluta original do preço não está disponível, mas o DTW
                    pós-normalização depende só do formato).

    `ref_date` / `lookback_months` recortam o escopo temporal antes de tudo
    (ver `_resolve_temporal_scope`): por default (ambos None) usa o período
    inteiro. O corte é aplicado sobre os retornos, então no modo "raw" o índice
    de preço é reconstruído já dentro da janela (base 100 no início do escopo).

    Ativos com menos de `min_periods` observações válidas são descartados.
    Séries têm comprimentos distintos (painel desbalanceado, IPOs pós-2019); o
    DTW do tslearn lida com isso nativamente via padding com NaN.

    Retorna (series_list, kept_assets, common_index) onde series_list é uma lista
    de np.ndarray 1D (uma por ativo) e common_index é o DatetimeIndex completo.
    """
    if series_mode not in ("returns", "raw"):
        raise ValueError(
            f"series_mode '{series_mode}' inválido (use 'returns' ou 'raw')."
        )

    clean_assets = [
        a for a in asset_list if a != "MARKET_INDEX" and a in master_data.columns
    ]

    start, end = _resolve_temporal_scope(master_data.index, ref_date, lookback_months)

    series_list = []
    kept_assets = []
    for asset in clean_assets:
        ret = _slice_temporal(master_data[asset].dropna(), start, end)
        if len(ret) < min_periods:
            continue
        if series_mode == "raw":
            level = 100.0 * (1.0 + ret / 100.0).cumprod()
            series = level
        else:
            series = ret
        series_list.append(series)
        kept_assets.append(asset)

    if len(kept_assets) == 0:
        raise ValueError(
            f"Nenhum ativo com >= {min_periods} observações em series_mode="
            f"'{series_mode}'."
        )

    common_index = master_data.index
    return series_list, kept_assets, common_index


def _normalize_dataset(X: np.ndarray, normalization: str) -> np.ndarray:
    """
    Normaliza o dataset de séries (n_assets x len x 1) conforme `normalization`.

    "meanvariance" : TimeSeriesScalerMeanVariance — z-norma CADA série de forma
                     independente (média 0, desvio 1). Invariante a amplitude:
                     só o formato da trajetória importa. É o scaler do exemplo do
                     tslearn.
    "zscore"       : z-score GLOBAL — um único μ/σ sobre todo o conjunto.
                     Preserva a amplitude relativa entre ativos (um ativo mais
                     volátil continua mais volátil), então o DTW também enxerga
                     diferença de escala, não só de formato.
    "none"         : sem normalização (séries cruas).

    NaN de padding (séries de comprimentos distintos) é preservado como NaN — o
    DTW do tslearn o trata como padding de comprimento variável.
    """
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    if normalization in (None, "none"):
        return X
    if normalization == "meanvariance":
        return TimeSeriesScalerMeanVariance().fit_transform(X)
    if normalization == "zscore":
        mu = np.nanmean(X)
        sigma = np.nanstd(X)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        return (X - mu) / sigma
    raise ValueError(
        f"normalization '{normalization}' inválido "
        "(use 'meanvariance', 'zscore' ou 'none')."
    )


def _silhouette_from_distances(
    dist_matrix: np.ndarray, raw_labels: np.ndarray, kept_assets: list
) -> tuple:
    """
    Silhouette médio + por ativo a partir de uma matriz de distâncias precomputada.

    Compartilhado pelos núcleos DTW e KShape para que a qualidade fique sempre na
    mesma escala [-1, 1] (é uma razão de distâncias), variando apenas a distância
    usada para medi-la: DTW/softdtw no DTW, Shape-Based Distance (SBD) no KShape.
    Mantém o sklearn `precomputed` para obter média e amostras por ativo de uma só
    matriz (a média == média das amostras).

    Retorna (labels Series Int64, sil_mean float, sil_samples Series).
    """
    from sklearn.metrics import silhouette_samples as _sk_sil_samples

    labels = pd.Series(raw_labels, index=kept_assets, name="cluster").astype("Int64")
    if len(set(raw_labels)) > 1:
        sil_mean = float(
            silhouette_score(dist_matrix, raw_labels, metric="precomputed")
        )
        sil_samples = _sk_sil_samples(dist_matrix, raw_labels, metric="precomputed")
    else:
        sil_mean = float("nan")
        sil_samples = np.full(len(raw_labels), np.nan)
    sil_samples_series = pd.Series(sil_samples, index=kept_assets, name="silhouette")
    return labels, sil_mean, sil_samples_series


def _run_dtw_clustering(
    value_list: list,
    kept_assets: list,
    n_clusters: int,
    normalization: str,
    metric: str,
    max_iter: int,
    random_state: int,
) -> dict:
    """
    Núcleo da clusterização DTW, agnóstico à origem das séries.

    `value_list` é uma lista de np.ndarray, um por ativo, de shape (T_i,) para
    séries univariadas ou (T_i, d) para multivariadas (mesmo `d` entre ativos,
    `T_i` livre). Centraliza fit do TimeSeriesKMeans, matriz de distâncias na
    métrica da clusterização (dtw ou divergência softdtw normalizada), silhouette
    (sklearn precomputed sobre essa matriz, em [-1, 1] como as demais funções) e
    barycenters. Compartilhado por `cluster_assets_dtw` (univariado) e
    `cluster_assets_dtw_signature` (multivariado: trajetória da assinatura macro).

    Retorna dict com raw_labels, labels, silhouette, silhouette_samples,
    barycenters (dict cluster -> ndarray (T, d)), inertia, dtw_distances e X
    (dataset já normalizado, para o caller reconstruir as séries no mesmo espaço
    dos barycenters).
    """
    from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_dtw, cdist_soft_dtw_normalized
    from tslearn.utils import to_time_series_dataset

    if len(kept_assets) < n_clusters:
        raise ValueError(
            f"Apenas {len(kept_assets)} ativos com séries válidas; "
            f"insuficiente para {n_clusters} clusters."
        )

    X = to_time_series_dataset(value_list)
    X = _normalize_dataset(X, normalization)

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        max_iter=max_iter,
        random_state=random_state,
        n_init=2,
    )
    raw_labels = model.fit_predict(X)

    if metric == "softdtw":
        dist_matrix = cdist_soft_dtw_normalized(X)
    else:
        dist_matrix = cdist_dtw(X)
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)  # sanea ruído numérico na diagonal
    np.clip(dist_matrix, 0.0, None, out=dist_matrix)

    labels, sil_mean, sil_samples_series = _silhouette_from_distances(
        dist_matrix, raw_labels, kept_assets
    )

    barycenters = {}
    for c in range(n_clusters):
        member_idx = np.where(raw_labels == c)[0]
        if len(member_idx) == 0:
            continue
        member_series = X[member_idx]
        if metric == "softdtw":
            bary = softdtw_barycenter(member_series)
        else:
            bary = dtw_barycenter_averaging(member_series)
        barycenters[c] = np.asarray(bary)  # (T, d)

    return {
        "raw_labels": raw_labels,
        "labels": labels,
        "silhouette": sil_mean,
        "silhouette_samples": sil_samples_series,
        "barycenters": barycenters,
        "inertia": float(model.inertia_),
        "dtw_distances": dist_matrix,
        "X": X,
    }


def _plot_silhouette_panel(
    ax,
    labels: pd.Series,
    sil_samples: pd.Series,
    sil_mean: float,
    n_clusters: int,
    colors: list,
    background_color: str,
    sil_metric_label: str = "DTW",
) -> None:
    """
    Desenha o painel de qualidade: barras horizontais do silhouette por ativo,
    agrupadas por cluster e ordenadas, com a linha tracejada do silhouette médio.
    Compartilhado pelos plots DTW e KShape; `sil_metric_label` apenas nomeia a
    distância no rótulo do eixo (DTW ou SBD).
    """
    ax.set_facecolor(background_color)
    x_lower = 0
    xticks = []
    present = []
    for c in range(n_clusters):
        members = labels[labels == c].index.tolist()
        vals = sil_samples.reindex(members).sort_values().dropna().values
        if len(vals) == 0:
            continue
        x_upper = x_lower + len(vals)
        ax.bar(
            range(x_lower, x_upper),
            vals,
            width=1.0,
            color=colors[c],
            edgecolor="none",
            alpha=0.85,
        )
        xticks.append((x_lower + x_upper) / 2)
        present.append(c)
        x_lower = x_upper + 2
    ax.axhline(
        sil_mean,
        color=styles.COLORS["sil_mean"],
        linestyle="--",
        linewidth=styles.lw("reference"),
        label=f"média={sil_mean:.3f}",
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"C{c}" for c in present])
    ax.set_ylabel(f"Silhouette ({sil_metric_label})", fontsize=9)
    ax.set_title("Qualidade por ativo", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def cluster_assets_dtw(
    master_data: pd.DataFrame,
    asset_list: list,
    n_clusters: int = 4,
    series_mode: str = "returns",
    min_periods: int = 24,
    normalization: str = "meanvariance",
    metric: str = "dtw",
    max_iter: int = 10,
    random_state: int = 42,
    ref_date=None,
    lookback_months=None,
) -> dict:
    """
    Clusteriza ativos pelo *formato* de suas séries temporais usando DTW
    (Dynamic Time Warping) com `tslearn.TimeSeriesKMeans`.

    Análogo a `cluster_assets_full_period`, mas o objeto clusterizado é a
    trajetória temporal do ativo — não o vetor de assinatura macro. Por isso a
    visualização é própria (`plot_dtw_clusters`): cada cluster vira um conjunto
    de séries com um barycenter (DBA) representativo no centro.

    Parameters
    ----------
    series_mode : str
        "returns" → série de retornos mensais; "raw" → índice de preço
        reconstruído (ver `_build_series_dataset`). É o parâmetro que alterna
        entre os dois modos de teste pedidos.
    min_periods : int
        Descarta ativos com menos observações válidas que isso.
    normalization : str
        Como escalar as séries antes do DTW (ver `_normalize_dataset`):
        "meanvariance" (default, scaler do exemplo do tslearn, invariante a
        amplitude), "zscore" (global, preserva amplitude relativa entre ativos)
        ou "none". Trocar entre "meanvariance" e "zscore" permite comparar se a
        amplitude relativa carrega sinal útil de clusterização.
    metric : str
        "dtw" (DBA barycenter) ou "softdtw" (soft-DTW barycenter).
    max_iter : int
        Iterações do k-means de séries temporais.
    ref_date, lookback_months :
        Recorte temporal opcional do escopo de clusterização. `ref_date` é o
        fim da janela (None → última data disponível) e `lookback_months` quanto
        olhar para trás a partir dele (None → desde o início). Por default
        (ambos None) a clusterização cobre todo o período.

    Returns
    -------
    dict com:
        labels             : Series (asset -> cluster)
        silhouette         : float — silhouette médio sobre a matriz DTW
                             (mesma escala do sil. das funções de assinatura)
        silhouette_samples : Series (asset -> silhouette individual)
        barycenters        : dict {cluster -> np.ndarray 1D} (centro DBA)
        series             : dict {asset -> pd.Series} séries usadas (já
                             normalizadas se normalize=True), indexadas por data
        inertia            : float — inércia DTW do modelo
        dtw_distances      : np.ndarray (n_assets x n_assets) matriz DTW
        asset_order        : list — ordem dos ativos nas matrizes
        n_clusters, series_mode, metric, normalization : metadados
    """
    series_list, kept_assets, common_index = _build_series_dataset(
        master_data, asset_list, series_mode, min_periods, ref_date, lookback_months
    )

    core = _run_dtw_clustering(
        [s.values for s in series_list],
        kept_assets,
        n_clusters,
        normalization,
        metric,
        max_iter,
        random_state,
    )

    # Séries de volta para pandas no espaço normalizado (mesmo dos barycenters),
    # alinhadas à própria janela temporal de cada ativo.
    series_dict = {}
    for asset, s, x_row in zip(kept_assets, series_list, core["X"]):
        values = np.asarray(x_row).ravel()[: len(s)]
        series_dict[asset] = pd.Series(values, index=s.index, name=asset)

    barycenters = {c: b.ravel() for c, b in core["barycenters"].items()}

    return {
        "labels": core["labels"],
        "silhouette": core["silhouette"],
        "silhouette_samples": core["silhouette_samples"],
        "barycenters": barycenters,
        "series": series_dict,
        "inertia": core["inertia"],
        "dtw_distances": core["dtw_distances"],
        "asset_order": kept_assets,
        "n_clusters": n_clusters,
        "series_mode": series_mode,
        "metric": metric,
        "normalization": normalization,
    }


def _plot_dtw_1d(
    result: dict,
    y_label: str,
    suptitle: str,
    ylim: tuple = None,
    draw_zero: bool = False,
    max_label_assets: int = 0,
    background_color: str = styles.BACKGROUND,
    sil_metric_label: str = "DTW",
) -> None:
    """
    Layout compartilhado dos plots de clusterização temporal univariados: um painel
    por linha (um por cluster) com as trajetórias dos ativos membros (cinza) e o
    centro do cluster (colorido — barycenter no DTW, shape extraction no KShape),
    mais uma linha final com o painel de qualidade (silhouette). Usado por
    `plot_dtw_clusters` / `plot_dtw_signature_clusters` (DTW) e por
    `plot_kshape_clusters` / `plot_kshape_signature_clusters` (KShape); cada wrapper
    só decide rótulo, título, limites de y, a linha do zero e o nome da distância
    do silhouette (`sil_metric_label`).

    `result["series"]` deve mapear asset -> pd.Series (1-D) e
    `result["barycenters"]` cluster -> np.ndarray 1-D.
    """
    labels = result["labels"]
    series = result["series"]
    barycenters = result["barycenters"]
    sil_samples = result["silhouette_samples"]
    sil_mean = result["silhouette"]
    n_clusters = result["n_clusters"]

    colors = styles.palette("clusters", n_clusters)

    sns.set_theme(style="whitegrid")
    n_rows = n_clusters + 1
    # Linha de qualidade mais alta que as de cluster (mais agradável com barras
    # verticais).
    quality_ratio = 1.8
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(13, 2.9 * (n_clusters + quality_ratio)),
        gridspec_kw={"height_ratios": [1] * n_clusters + [quality_ratio]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor(background_color)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for c in range(n_clusters):
        ax = axes[c]
        ax.set_facecolor(background_color)
        members = labels[labels == c].index.tolist()
        for asset in members:
            s = series[asset]
            ax.plot(
                s.index,
                s.values,
                color=styles.COLORS["member"],
                alpha=0.35,
                linewidth=styles.lw("member"),
            )
        if c in barycenters:
            bary = barycenters[c]
            # eixo x do barycenter: usa a série membro mais longa como referência
            longest = max(members, key=lambda a: len(series[a])) if members else None
            if longest is not None:
                x_ref = series[longest].index[: len(bary)]
                ax.plot(
                    x_ref,
                    bary[: len(x_ref)],
                    color=colors[c],
                    linewidth=styles.lw("barycenter"),
                    label="barycenter",
                )
                ax.legend(loc="upper right", fontsize=8, frameon=False)
        if draw_zero:
            ax.axhline(
                0,
                color=styles.COLORS["ref_line"],
                linewidth=styles.lw("reference"),
                linestyle="--",
                alpha=0.6,
            )
        if ylim is not None:
            ax.set_ylim(*ylim)
        if max_label_assets and members:
            ordered = sil_samples.reindex(members).sort_values(ascending=False)
            for asset in ordered.index[:max_label_assets]:
                s = series[asset]
                ax.text(
                    s.index[-1],
                    s.values[-1],
                    asset,
                    fontsize=6.5,
                    color=colors[c],
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                )
        cluster_sil = sil_samples.reindex(members).mean()
        ax.set_title(
            f"Cluster {c}  (n={len(members)})  ·  sil={cluster_sil:.3f}",
            fontsize=11,
            loc="left",
        )
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.set_ylabel(y_label, fontsize=9)

    _plot_silhouette_panel(
        axes[-1],
        labels,
        sil_samples,
        sil_mean,
        n_clusters,
        colors,
        background_color,
        sil_metric_label=sil_metric_label,
    )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.show()


def plot_dtw_clusters(
    result: dict,
    max_label_assets: int = 0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Visualiza a clusterização DTW (retorno/preço): um painel por cluster com as
    trajetórias dos ativos membros (cinza) e o barycenter (colorido), mais um
    painel de silhouette por ativo — a medida de qualidade comparável ao
    silhouette das funções de assinatura.

    max_label_assets : int
        Se > 0, anota o nome de até esse nº de ativos por cluster (os mais
        próximos do barycenter). 0 = sem anotação (default, evita poluição).
    """
    series_mode = result["series_mode"]
    normalization = result["normalization"]
    metric = result["metric"]
    sil_mean = result["silhouette"]

    base_label = "Retorno mensal" if series_mode == "returns" else "Índice de preço"
    if normalization in (None, "none"):
        unit = "%" if series_mode == "returns" else ""
    else:
        unit = "z"
    y_label = f"{base_label} ({unit})".replace(" ()", "")

    suptitle = (
        f"Clusterização DTW ({metric}, {series_mode}, norm={normalization})  |  "
        f"Silhouette médio: {sil_mean:.3f}"
    )
    _plot_dtw_1d(
        result,
        y_label,
        suptitle,
        ylim=None,
        draw_zero=False,
        max_label_assets=max_label_assets,
        background_color=background_color,
    )


# =============================================================================
# Camada clustering/ — DTW sobre a trajetória da ASSINATURA macro (1 IMec-guia)
#
# Mescla as duas ideias: em vez de clusterizar a série de retornos/preço, o
# objeto é a TRAJETÓRIA da rolling correlation do ativo contra UM IMec-guia
# (`target_macro`) — uma série temporal 1-D que descreve como a sensibilidade do
# ativo àquele macro evolui no tempo. O DTW alinha trajetórias defasadas e o
# silhouette (matriz DTW) mede a qualidade na mesma escala das demais funções.
# Como é univariado, a visualização reaproveita o layout do `plot_dtw_clusters`.
# =============================================================================


def _build_signature_series_dataset(
    master_data: pd.DataFrame,
    asset_list: list,
    target_macro: str,
    window_size: int,
    correlation_method: str,
    lag: int,
    min_periods: int,
    ref_date=None,
    lookback_months=None,
) -> tuple:
    """
    Monta, por ativo, a trajetória 1-D da correlação móvel contra `target_macro`,
    a partir da rolling correlation cacheada (`calculate_rolling_correlations`,
    target asset-macro). Cada ativo vira uma série temporal: em cada janela, a
    correlação do ativo com o IMec-guia.

    A rolling correlation já vem com o forward-fill por ativo aplicado; aqui só
    removemos NaN remanescente (início da série). `ref_date` / `lookback_months`
    recortam o escopo temporal das trajetórias (ver `_resolve_temporal_scope`);
    por default (ambos None) usa o período inteiro. Ativos com < min_periods
    janelas (após o recorte) são descartados.

    Retorna (value_list, index_list, kept_assets): value_list é a lista de
    np.ndarray 1-D; index_list os DatetimeIndex correspondentes.
    """
    clean_assets = [a for a in asset_list if a != "MARKET_INDEX"]

    rolling_corr_df = calculate_rolling_correlations(
        master_data,
        clean_assets,
        window_size=window_size,
        method=correlation_method,
        target="asset-macro",
        lag=lag,
        lag_features=[target_macro],
    )

    if target_macro not in rolling_corr_df.columns:
        raise ValueError(
            f"target_macro '{target_macro}' não está nas features disponíveis "
            f"({list(rolling_corr_df.columns)})."
        )

    available_assets = set(rolling_corr_df.index.get_level_values(1).unique())

    start, end = _resolve_temporal_scope(
        rolling_corr_df.index.get_level_values(0), ref_date, lookback_months
    )

    value_list = []
    index_list = []
    kept_assets = []
    for asset in clean_assets:
        if asset not in available_assets:
            continue
        traj = rolling_corr_df.xs(asset, level=1)[target_macro].dropna().sort_index()
        traj = _slice_temporal(traj, start, end)
        if len(traj) < min_periods:
            continue
        value_list.append(traj.values)
        index_list.append(traj.index)
        kept_assets.append(asset)

    if len(kept_assets) == 0:
        raise ValueError(
            f"Nenhum ativo com >= {min_periods} janelas de assinatura válidas "
            f"(window_size={window_size}, target_macro={target_macro})."
        )

    return value_list, index_list, kept_assets


def cluster_assets_dtw_signature(
    master_data: pd.DataFrame,
    asset_list: list,
    target_macro: str,
    n_clusters: int = 4,
    window_size: int = 6,
    lag: int = 0,
    correlation_method: str = "spearman",
    min_periods: int = 12,
    normalization: str = "none",
    metric: str = "dtw",
    max_iter: int = 10,
    random_state: int = 42,
    ref_date=None,
    lookback_months=None,
) -> dict:
    """
    Clusteriza ativos por DTW sobre a trajetória da correlação móvel contra um
    único IMec-guia (`target_macro`).

    Mescla `cluster_assets_dtw` (DTW sobre séries temporais) com a ideia de
    assinatura: o objeto clusterizado é a série temporal 1-D de como o ativo se
    correlaciona com `target_macro` ao longo das janelas. Dois ativos cuja
    sensibilidade àquele macro evolui de forma parecida — mesmo defasados no
    tempo — caem no mesmo cluster. Guiar por um único IMec tende a dar clusters
    mais nítidos que a versão multivariada (todos os IMecs de uma vez).

    Parameters
    ----------
    target_macro : str
        IMec-guia da clusterização (ex.: "macro_dollar_var"). Deve ser uma das
        colunas macro da rolling correlation.
    window_size : int
        Janela (meses) da rolling correlation que gera a assinatura.
    correlation_method : str
        Método da correlação ("spearman", "pearson", "dcor", ...).
    min_periods : int
        Mínimo de janelas válidas para o ativo entrar na clusterização.
    normalization : str
        "none" (default) preserva a magnitude da correlação — o próprio sinal de
        sensibilidade macro, já em [-1, 1]. "meanvariance" foca só no formato
        temporal; "zscore" é z-score global. Ver `_normalize_dataset`.
    ref_date, lookback_months :
        Recorte temporal opcional das trajetórias de correlação. `ref_date` é o
        fim da janela (None → última data disponível) e `lookback_months` quanto
        olhar para trás (None → desde o início). Por default (ambos None) a
        clusterização cobre todo o período.

    Returns
    -------
    dict com a mesma espinha dos outros resultados DTW (univariado):
        labels, silhouette, silhouette_samples, inertia, dtw_distances,
        asset_order, n_clusters, metric, normalization, correlation_method,
        window_size
        barycenters : dict {cluster -> np.ndarray 1-D}
        series      : dict {asset  -> pd.Series} no espaço normalizado (mesmo
                      dos barycenters), indexadas por data
        series_mode : "signature"
        target_macro: o IMec-guia usado
    """
    value_list, index_list, kept_assets = _build_signature_series_dataset(
        master_data,
        asset_list,
        target_macro,
        window_size,
        correlation_method,
        lag,
        min_periods,
        ref_date,
        lookback_months,
    )

    core = _run_dtw_clustering(
        value_list,
        kept_assets,
        n_clusters,
        normalization,
        metric,
        max_iter,
        random_state,
    )

    # Séries normalizadas de volta para pandas (mesmo espaço dos barycenters).
    series_dict = {}
    for asset, x_row, idx in zip(kept_assets, core["X"], index_list):
        values = np.asarray(x_row).ravel()[: len(idx)]
        series_dict[asset] = pd.Series(values, index=idx, name=asset)

    barycenters = {c: b.ravel() for c, b in core["barycenters"].items()}

    return {
        "labels": core["labels"],
        "silhouette": core["silhouette"],
        "silhouette_samples": core["silhouette_samples"],
        "barycenters": barycenters,
        "series": series_dict,
        "inertia": core["inertia"],
        "dtw_distances": core["dtw_distances"],
        "asset_order": kept_assets,
        "n_clusters": n_clusters,
        "series_mode": "signature",
        "metric": metric,
        "normalization": normalization,
        "correlation_method": correlation_method,
        "window_size": window_size,
        "target_macro": target_macro,
    }


def plot_dtw_signature_clusters(
    result: dict,
    max_label_assets: int = 0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Visualiza a clusterização DTW da assinatura guiada por um IMec.

    Mesmo layout do `plot_dtw_clusters` (um painel por cluster com as trajetórias
    de correlação dos membros + barycenter, e o painel de silhouette à direita),
    mas o eixo y é a correlação móvel com o `target_macro` (limites [-1, 1] ou
    [0, 1] para dcor, com linha do zero).
    """
    metric = result["metric"]
    normalization = result["normalization"]
    sil_mean = result["silhouette"]
    corr_method = result.get("correlation_method", "spearman")
    window_size = result.get("window_size", "?")
    target_macro = result["target_macro"]

    raw_scale = normalization in (None, "none")
    if raw_scale:
        ylim = (-0.05, 1.05) if corr_method == "dcor" else (-1.1, 1.1)
    else:
        ylim = None

    macro_short = MACRO_TITLES.get(target_macro, target_macro).split("(")[0].strip()
    y_label = f"Corr. {macro_short}" if raw_scale else f"Corr. {macro_short} (z)"

    suptitle = (
        f"Clusterização DTW da assinatura — {macro_short}  |  "
        f"W{window_size} {corr_method}, {metric}, norm={normalization}  |  "
        f"Silhouette médio: {sil_mean:.3f}"
    )
    _plot_dtw_1d(
        result,
        y_label,
        suptitle,
        ylim=ylim,
        draw_zero=raw_scale,
        max_label_assets=max_label_assets,
        background_color=background_color,
    )


# =============================================================================
# Camada clustering/ — clusterização por KShape sobre as séries temporais
#
# Segundo backend shape-based da taxonomia Paparrizos (depois do DTW). O KShape
# (Paparrizos & Gravano, SIGMOD 2015) agrupa séries por CORRELAÇÃO CRUZADA
# normalizada: é invariante a deslocamento de fase (alinha picos defasados) e a
# amplitude, capturando o *formato* da trajetória. Diferente do DTW — que aceita
# séries de tamanhos distintos via padding NaN — o KShape baseia-se em FFT e
# EXIGE séries de MESMO comprimento e sem NaN; por isso recortamos cada série à
# sua cauda mais recente de tamanho fixo (`_crop_to_equal_length`). A qualidade é
# medida por um silhouette sobre a matriz de Shape-Based Distance (SBD =
# 1 - max NCC), na MESMA escala [-1, 1] do silhouette do DTW e das assinaturas —
# direto comparável. Como o objeto e o resultado têm a mesma forma do DTW
# univariado, a visualização reaproveita `_plot_dtw_1d` via wrappers finos.
# =============================================================================


def _crop_to_equal_length(
    value_list: list, index_list: list, kept_assets: list, length: int
) -> tuple:
    """
    Recorta cada série à sua cauda mais recente de `length` pontos e descarta as
    mais curtas.

    O KShape exige séries de mesmo comprimento (correlação cruzada por FFT), ao
    contrário do DTW (que alinha tamanhos distintos via padding NaN). Igualamos
    todas pegando os últimos `length` pontos de cada uma; ativos com menos de
    `length` observações saem da clusterização. Como o KShape é invariante a
    deslocamento, usar a cauda recente de cada ativo (sem exigir alinhamento de
    calendário) é coerente com o método e preserva o painel desbalanceado.

    Retorna (value_list, index_list, kept_assets) já recortados e alinhados.
    """
    out_vals, out_idx, out_assets = [], [], []
    for values, idx, asset in zip(value_list, index_list, kept_assets):
        values = np.asarray(values)
        if len(values) < length:
            continue
        out_vals.append(values[-length:])
        out_idx.append(idx[-length:])
        out_assets.append(asset)
    if len(out_assets) == 0:
        raise ValueError(
            f"Nenhum ativo com >= {length} observações para o KShape "
            "(que exige séries de mesmo comprimento)."
        )
    return out_vals, out_idx, out_assets


def _run_kshape_clustering(
    value_list: list,
    kept_assets: list,
    n_clusters: int,
    normalization: str,
    max_iter: int,
    random_state: int,
) -> dict:
    """
    Núcleo da clusterização KShape, agnóstico à origem das séries.

    Análogo a `_run_dtw_clustering`, mas o agrupador é `tslearn.clustering.KShape`
    (correlação cruzada normalizada, invariante a fase). `value_list` deve já vir
    recortado a um comprimento comum (`_crop_to_equal_length`) — o KShape rejeita
    NaN/comprimentos distintos. Centraliza fit do KShape, matriz de Shape-Based
    Distance (SBD = 1 - max NCC), silhouette (sklearn precomputed sobre o SBD, em
    [-1, 1] como as demais funções) e os centroides de shape extraction. Compartilhado
    por `cluster_assets_kshape` (univariado) e `cluster_assets_kshape_signature`.

    Retorna dict com raw_labels, labels, silhouette, silhouette_samples,
    barycenters (dict cluster -> centro KShape ndarray (T, d)), inertia,
    distances (matriz SBD) e X (dataset já normalizado).
    """
    from tslearn.clustering import KShape
    from tslearn.metrics import cdist_normalized_cc
    from tslearn.utils import to_time_series_dataset

    if len(kept_assets) < n_clusters:
        raise ValueError(
            f"Apenas {len(kept_assets)} ativos com séries válidas; "
            f"insuficiente para {n_clusters} clusters."
        )

    X = to_time_series_dataset(value_list)
    if np.isnan(X).any():
        raise ValueError(
            "KShape exige séries de mesmo comprimento (sem NaN). Recorte-as a um "
            "comprimento comum antes (ver _crop_to_equal_length)."
        )
    X = _normalize_dataset(X, normalization)

    model = KShape(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init=2,
    )
    raw_labels = model.fit_predict(X)

    norms = np.linalg.norm(X, axis=(1, 2))
    cc = cdist_normalized_cc(X, X, norms, norms, self_similarity=False)
    dist_matrix = 1.0 - np.asarray(cc, dtype=float)
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)  # sanea ruído numérico ~1e-16
    np.fill_diagonal(dist_matrix, 0.0)
    np.clip(dist_matrix, 0.0, None, out=dist_matrix)

    labels, sil_mean, sil_samples_series = _silhouette_from_distances(
        dist_matrix, raw_labels, kept_assets
    )

    barycenters = {}
    for c in range(n_clusters):
        if (raw_labels == c).any():
            barycenters[c] = np.asarray(model.cluster_centers_[c])  # (T, d)

    return {
        "raw_labels": raw_labels,
        "labels": labels,
        "silhouette": sil_mean,
        "silhouette_samples": sil_samples_series,
        "barycenters": barycenters,
        "inertia": float(getattr(model, "inertia_", np.nan)),
        "distances": dist_matrix,
        "X": X,
    }


def cluster_assets_kshape(
    master_data: pd.DataFrame,
    asset_list: list,
    n_clusters: int = 4,
    series_mode: str = "returns",
    series_length: int = 36,
    normalization: str = "meanvariance",
    max_iter: int = 100,
    random_state: int = 42,
    ref_date=None,
) -> dict:
    """
    Clusteriza ativos pelo *formato* de suas séries temporais usando KShape
    (Paparrizos & Gravano, 2015), baseado em correlação cruzada normalizada.

    Contraparte do `cluster_assets_dtw`: mesmo objeto (a trajetória de
    retorno/preço do ativo) e mesma forma de resultado, mas o agrupamento é por
    KShape em vez de DTW. O KShape é invariante a deslocamento de fase e a
    amplitude, então agrupa ativos com o mesmo formato de trajetória mesmo que
    defasados — porém EXIGE séries de mesmo comprimento, então cada ativo é
    recortado aos seus últimos `series_length` meses (ver `_crop_to_equal_length`)
    e ativos com histórico menor são descartados.

    Parameters
    ----------
    series_mode : str
        "returns" → série de retornos mensais; "raw" → índice de preço
        reconstruído (ver `_build_series_dataset`).
    series_length : int
        Comprimento comum (em meses) das séries: cada ativo usa seus últimos
        `series_length` pontos. É o knob que troca cobertura de ativos por riqueza
        de formato (mais longo = menos ativos elegíveis, formato mais detalhado).
    normalization : str
        Escalonamento antes do KShape (ver `_normalize_dataset`). Default
        "meanvariance" (z-norma por série), como recomenda o exemplo do tslearn —
        o KShape pressupõe séries escaladas. "zscore"/"none" também aceitos.
    max_iter : int
        Iterações do KShape.
    ref_date :
        Fim do recorte temporal (None → última data disponível). A cauda de
        `series_length` meses é tomada até `ref_date`.

    Returns
    -------
    dict com a mesma espinha dos resultados DTW univariados:
        labels, silhouette, silhouette_samples, barycenters (dict cluster ->
        np.ndarray 1-D, o centro KShape), series (dict asset -> pd.Series no
        espaço normalizado), inertia, distances (matriz SBD), asset_order,
        n_clusters, series_mode, series_length, normalization.
    """
    series_list, kept_assets, _ = _build_series_dataset(
        master_data,
        asset_list,
        series_mode,
        min_periods=series_length,
        ref_date=ref_date,
        lookback_months=None,
    )
    value_list = [s.values for s in series_list]
    index_list = [s.index for s in series_list]
    value_list, index_list, kept_assets = _crop_to_equal_length(
        value_list, index_list, kept_assets, series_length
    )

    core = _run_kshape_clustering(
        value_list, kept_assets, n_clusters, normalization, max_iter, random_state
    )

    # Séries de volta para pandas no espaço normalizado (mesmo dos centros KShape).
    series_dict = {}
    for asset, x_row, idx in zip(kept_assets, core["X"], index_list):
        values = np.asarray(x_row).ravel()[: len(idx)]
        series_dict[asset] = pd.Series(values, index=idx, name=asset)

    barycenters = {c: b.ravel() for c, b in core["barycenters"].items()}

    return {
        "labels": core["labels"],
        "silhouette": core["silhouette"],
        "silhouette_samples": core["silhouette_samples"],
        "barycenters": barycenters,
        "series": series_dict,
        "inertia": core["inertia"],
        "distances": core["distances"],
        "asset_order": kept_assets,
        "n_clusters": n_clusters,
        "series_mode": series_mode,
        "series_length": series_length,
        "normalization": normalization,
    }


def plot_kshape_clusters(
    result: dict,
    max_label_assets: int = 0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Visualiza a clusterização KShape (retorno/preço) reaproveitando o layout do
    `plot_dtw_clusters` (`_plot_dtw_1d`): um painel por cluster com as trajetórias
    dos ativos membros (cinza) e o centro KShape (shape extraction, colorido), mais
    o painel de silhouette por ativo — aqui medido na distância SBD.
    """
    series_mode = result["series_mode"]
    normalization = result["normalization"]
    sil_mean = result["silhouette"]
    series_length = result.get("series_length", "?")

    base_label = "Retorno mensal" if series_mode == "returns" else "Índice de preço"
    if normalization in (None, "none"):
        unit = "%" if series_mode == "returns" else ""
    else:
        unit = "z"
    y_label = f"{base_label} ({unit})".replace(" ()", "")

    suptitle = (
        f"Clusterização KShape ({series_mode}, L={series_length}, "
        f"norm={normalization})  |  Silhouette médio (SBD): {sil_mean:.3f}"
    )
    _plot_dtw_1d(
        result,
        y_label,
        suptitle,
        ylim=None,
        draw_zero=False,
        max_label_assets=max_label_assets,
        background_color=background_color,
        sil_metric_label="SBD",
    )


def cluster_assets_kshape_signature(
    master_data: pd.DataFrame,
    asset_list: list,
    target_macro: str,
    n_clusters: int = 4,
    window_size: int = 6,
    lag: int = 0,
    correlation_method: str = "spearman",
    series_length: int = 24,
    normalization: str = "meanvariance",
    max_iter: int = 100,
    random_state: int = 42,
    ref_date=None,
) -> dict:
    """
    Clusteriza ativos por KShape sobre a trajetória da correlação móvel contra um
    único IMec-guia (`target_macro`).

    Contraparte KShape do `cluster_assets_dtw_signature`: o objeto clusterizado é a
    série temporal 1-D de como o ativo se correlaciona com `target_macro` ao longo
    das janelas. Dois ativos cuja sensibilidade àquele macro evolui com o mesmo
    formato — mesmo defasados — caem no mesmo cluster. Como o KShape exige
    comprimento comum, cada trajetória é recortada aos seus últimos `series_length`
    pontos e ativos com histórico menor são descartados.

    Parameters
    ----------
    target_macro : str
        IMec-guia (ex.: "macro_dollar_var"); coluna macro da rolling correlation.
    window_size : int
        Janela (meses) da rolling correlation que gera a assinatura.
    correlation_method : str
        Método da correlação ("spearman", "pearson", "dcor", ...).
    series_length : int
        Comprimento comum (em janelas) das trajetórias de correlação.
    normalization : str
        Escalonamento antes do KShape (ver `_normalize_dataset`). Default
        "meanvariance"; "none" preserva a escala [-1, 1] da correlação.
    ref_date :
        Fim do recorte temporal das trajetórias (None → última data disponível).

    Returns
    -------
    dict com a mesma espinha dos resultados DTW univariados, mais
        series_mode="signature", target_macro, window_size, correlation_method,
        series_length.
    """
    value_list, index_list, kept_assets = _build_signature_series_dataset(
        master_data,
        asset_list,
        target_macro,
        window_size,
        correlation_method,
        lag,
        min_periods=series_length,
        ref_date=ref_date,
        lookback_months=None,
    )
    value_list, index_list, kept_assets = _crop_to_equal_length(
        value_list, index_list, kept_assets, series_length
    )

    core = _run_kshape_clustering(
        value_list, kept_assets, n_clusters, normalization, max_iter, random_state
    )

    series_dict = {}
    for asset, x_row, idx in zip(kept_assets, core["X"], index_list):
        values = np.asarray(x_row).ravel()[: len(idx)]
        series_dict[asset] = pd.Series(values, index=idx, name=asset)

    barycenters = {c: b.ravel() for c, b in core["barycenters"].items()}

    return {
        "labels": core["labels"],
        "silhouette": core["silhouette"],
        "silhouette_samples": core["silhouette_samples"],
        "barycenters": barycenters,
        "series": series_dict,
        "inertia": core["inertia"],
        "distances": core["distances"],
        "asset_order": kept_assets,
        "n_clusters": n_clusters,
        "series_mode": "signature",
        "normalization": normalization,
        "correlation_method": correlation_method,
        "window_size": window_size,
        "series_length": series_length,
        "target_macro": target_macro,
    }


def plot_kshape_signature_clusters(
    result: dict,
    max_label_assets: int = 0,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Visualiza a clusterização KShape da assinatura guiada por um IMec.

    Mesmo layout do `plot_dtw_signature_clusters` (`_plot_dtw_1d`), com o eixo y na
    correlação móvel com `target_macro` (limites [-1, 1] ou [0, 1] para dcor quando
    a normalização preserva a escala), e o silhouette medido na distância SBD.
    """
    normalization = result["normalization"]
    sil_mean = result["silhouette"]
    corr_method = result.get("correlation_method", "spearman")
    window_size = result.get("window_size", "?")
    target_macro = result["target_macro"]

    raw_scale = normalization in (None, "none")
    if raw_scale:
        ylim = (-0.05, 1.05) if corr_method == "dcor" else (-1.1, 1.1)
    else:
        ylim = None

    macro_short = MACRO_TITLES.get(target_macro, target_macro).split("(")[0].strip()
    y_label = f"Corr. {macro_short}" if raw_scale else f"Corr. {macro_short} (z)"

    suptitle = (
        f"Clusterização KShape da assinatura — {macro_short}  |  "
        f"W{window_size} {corr_method}, norm={normalization}  |  "
        f"Silhouette médio (SBD): {sil_mean:.3f}"
    )
    _plot_dtw_1d(
        result,
        y_label,
        suptitle,
        ylim=ylim,
        draw_zero=raw_scale,
        max_label_assets=max_label_assets,
        background_color=background_color,
        sil_metric_label="SBD",
    )


def plot_clustering_overview(
    result: dict,
    macro_features: list,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Plota: (i) heatmap cluster x IMEC dos centroides modais e (ii) grid 2x2 com
    a evolução temporal de cada coordenada do centroide ao longo das janelas.
    """
    centroids = result["centroids"]
    centroids_per_window = result["centroids_per_window"]
    sil_mean = result["sil_mean"]
    flip_rate_mean = result["flip_rate_mean"]
    ref = result["ref_date"]
    method = result.get("correlation_method", "spearman")
    n_clusters = len(centroids)

    is_dcor = method == "dcor"
    heat_vmin, heat_vmax = (0.0, 1.0) if is_dcor else (-1.0, 1.0)
    heat_center = None if is_dcor else 0.0
    heat_cmap = styles.heatmap_cmap(method)
    line_ylim = (-0.05, 1.05) if is_dcor else (-1.1, 1.1)

    evo_records = []
    for dt, c_df in centroids_per_window.items():
        for cluster_id, row in c_df.iterrows():
            for feat in macro_features:
                evo_records.append(
                    {
                        "date": dt,
                        "cluster": int(cluster_id),
                        "feature": feat,
                        "value": row[feat],
                    }
                )
    evo_df = pd.DataFrame(evo_records)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor(background_color)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.3, 1, 1], hspace=0.45, wspace=0.3)

    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(background_color)
    display_centroids = centroids.copy()
    display_centroids.columns = [
        MACRO_TITLES.get(c, c).split("(")[0].strip() for c in display_centroids.columns
    ]
    sns.heatmap(
        display_centroids,
        annot=True,
        cmap=heat_cmap,
        center=heat_center,
        vmin=heat_vmin,
        vmax=heat_vmax,
        fmt=".2f",
        linewidths=0.5,
        ax=ax_heat,
        cbar_kws={"label": f"Correlação média ({method})"},
    )
    ax_heat.set_title("Perfil dos clusters (centroides modais)", fontsize=13)
    ax_heat.set_ylabel("Cluster")
    ax_heat.set_xlabel("")
    ax_heat.tick_params(axis="x", rotation=30)

    colors = styles.palette("clusters", n_clusters)
    evo_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax, feat in zip(evo_axes, macro_features):
        ax.set_facecolor(background_color)
        sub = evo_df[evo_df["feature"] == feat]
        for cluster_id in sorted(sub["cluster"].unique()):
            line = sub[sub["cluster"] == cluster_id].sort_values("date")
            ax.plot(
                line["date"],
                line["value"],
                label=f"C{cluster_id}",
                color=colors[cluster_id],
                linewidth=styles.lw("evolution"),
            )
        if not is_dcor:
            ax.axhline(
                0,
                color=styles.COLORS["ref_line"],
                linewidth=styles.lw("reference"),
                linestyle="--",
                alpha=0.7,
            )
        ax.set_ylim(*line_ylim)
        ax.set_title(MACRO_TITLES.get(feat, feat), fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylabel("Correlação", fontsize=9)

    evo_axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.05, 1.28),
        ncol=n_clusters,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        f"Clusterização por janela  |  Ref: {ref.strftime('%Y-%m')}  |  "
        f"Silhouette médio: {sil_mean:.3f}  |  Flip rate médio: {flip_rate_mean:.3f}",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )
    plt.show()


def sweep_k(
    master_data: pd.DataFrame,
    asset_list: list,
    macro_features: list,
    k_values=range(2, 7),
    window_size: int = 6,
    lag: int = 0,
    ref_date: str = None,
    lookback_months: int = None,
    cluster_method: str = "kmeans",
    correlation_method: str = "spearman",
    exclude_features: list = None,
) -> pd.DataFrame:
    """
    Varre n_clusters em k_values, mantendo o restante fixo, e retorna um
    DataFrame indexado por k com sil_mean e flip_rate_mean.
    """
    rows = []
    for k in k_values:
        result = cluster_assets_per_window(
            master_data,
            asset_list,
            macro_features,
            n_clusters=k,
            window_size=window_size,
            lag=lag,
            ref_date=ref_date,
            lookback_months=lookback_months,
            cluster_method=cluster_method,
            correlation_method=correlation_method,
            exclude_features=exclude_features,
        )
        rows.append(
            {
                "k": k,
                "sil_mean": result["sil_mean"],
                "flip_rate_mean": result["flip_rate_mean"],
            }
        )
    return pd.DataFrame(rows).set_index("k")


def plot_k_sweep(
    sweep_df: pd.DataFrame,
    background_color: str = styles.BACKGROUND,
) -> None:
    """Plota silhouette médio e flip rate médio sobre k em eixos gêmeos."""
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(background_color)
    ax1.set_facecolor(background_color)

    sil_color = styles.KSWEEP["silhouette"]
    flip_color = styles.KSWEEP["flip"]

    ax1.plot(
        sweep_df.index,
        sweep_df["sil_mean"],
        marker="o",
        color=sil_color,
        linewidth=styles.lw("primary"),
        label="Silhouette médio",
    )
    ax1.set_xlabel("k (n_clusters)")
    ax1.set_ylabel("Silhouette médio", color=sil_color)
    ax1.tick_params(axis="y", labelcolor=sil_color)
    ax1.set_xticks(list(sweep_df.index))

    ax2 = ax1.twinx()
    ax2.plot(
        sweep_df.index,
        sweep_df["flip_rate_mean"],
        marker="s",
        color=flip_color,
        linewidth=styles.lw("primary"),
        linestyle="--",
        label="Flip rate médio",
    )
    ax2.set_ylabel("Flip rate médio", color=flip_color)
    ax2.tick_params(axis="y", labelcolor=flip_color)
    ax2.grid(False)

    ax1.set_title(
        "Sweep de k — silhouette vs flip rate", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def plot_clusters_3d(
    result: dict,
    annotate: bool = False,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Scatter 3D dos ativos em coordenadas de macro_signature média (no lookback),
    colorido pelo cluster modal. Centroides modais como marcadores grandes (X).
    Requer exatamente 3 macro_features no resultado.
    """
    asset_signature = result["asset_signature"]
    modal_label = result["modal_label"]
    centroids = result["centroids"]
    features = asset_signature.columns.tolist()

    if len(features) != 3:
        raise ValueError(
            f"plot_clusters_3d requer 3 macro_features; recebeu {len(features)}. "
            f"Use exclude_features na clusterização para chegar em 3."
        )

    method = result.get("correlation_method", "spearman")
    n_clusters = len(centroids)
    sil_mean = result["sil_mean"]
    flip_rate_mean = result["flip_rate_mean"]
    ref = result["ref_date"]

    colors = styles.palette("clusters", n_clusters)
    short_labels = [MACRO_TITLES.get(f, f).split("(")[0].strip() for f in features]

    data_for_lims = pd.concat([asset_signature, centroids], axis=0)
    feat_span = (data_for_lims.max() - data_for_lims.min()).replace(0, 0.1)
    pad = feat_span * 0.08
    lims_by_feat = {
        f: (data_for_lims[f].min() - pad[f], data_for_lims[f].max() + pad[f])
        for f in features
    }

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor(background_color)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(background_color)

    for cluster_id in range(n_clusters):
        members = modal_label[modal_label == cluster_id].index
        sub = asset_signature.loc[asset_signature.index.intersection(members)]
        if sub.empty:
            continue
        ax.scatter(
            sub[features[0]],
            sub[features[1]],
            sub[features[2]],
            color=colors[cluster_id],
            s=55,
            alpha=0.75,
            edgecolors="white",
            linewidth=0.5,
            label=f"C{cluster_id}  (n={len(sub)})",
        )
        if annotate:
            for asset, row in sub.iterrows():
                ax.text(
                    row[features[0]],
                    row[features[1]],
                    row[features[2]],
                    asset,
                    fontsize=7,
                    color="black",
                )

    for cluster_id, row in centroids.iterrows():
        ax.scatter(
            row[features[0]],
            row[features[1]],
            row[features[2]],
            color=colors[int(cluster_id)],
            s=350,
            marker="X",
            edgecolors="black",
            linewidth=1.5,
        )

    ax.set_xlim(*lims_by_feat[features[0]])
    ax.set_ylim(*lims_by_feat[features[1]])
    ax.set_zlim(*lims_by_feat[features[2]])
    ax.set_xlabel(short_labels[0], fontsize=9, labelpad=8)
    ax.set_ylabel(short_labels[1], fontsize=9, labelpad=8)
    ax.set_zlabel(short_labels[2], fontsize=9, labelpad=8)

    ax.set_title(
        f"Clusters em 3D ({method})  |  Ref: {ref.strftime('%Y-%m')}\n"
        f"Silhouette médio: {sil_mean:.3f}  |  Flip rate médio: {flip_rate_mean:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_clusters_pairs(
    result: dict,
    annotate: bool = True,
    background_color: str = styles.BACKGROUND,
) -> None:
    """
    Pairs plot: para cada combinação de 2 macro_features, um scatter 2D dos
    ativos coloridos pelo cluster modal, com os centroides modais sobrepostos
    como marcadores X. Para 4 features → 6 painéis em grid 2x3.
    """
    from itertools import combinations

    asset_signature = result["asset_signature"]
    modal_label = result["modal_label"]
    centroids = result["centroids"]
    features = asset_signature.columns.tolist()
    n_features = len(features)

    if n_features < 2:
        raise ValueError("plot_clusters_pairs precisa de pelo menos 2 features.")

    method = result.get("correlation_method", "spearman")
    n_clusters = len(centroids)
    sil_mean = result["sil_mean"]
    flip_rate_mean = result["flip_rate_mean"]
    ref = result["ref_date"]

    is_dcor = method == "dcor"

    data_for_lims = pd.concat([asset_signature, centroids], axis=0)
    feat_span = (data_for_lims.max() - data_for_lims.min()).replace(0, 0.1)
    pad = feat_span * 0.08
    lims_by_feat = {
        f: (data_for_lims[f].min() - pad[f], data_for_lims[f].max() + pad[f])
        for f in features
    }

    pairs = list(combinations(range(n_features), 2))
    n_pairs = len(pairs)
    cols = min(3, n_pairs)
    rows = int(np.ceil(n_pairs / cols))

    short_labels = [MACRO_TITLES.get(f, f).split("(")[0].strip() for f in features]
    colors = styles.palette("clusters", n_clusters)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.3))
    fig.patch.set_facecolor(background_color)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    cluster_handles = {}

    for ax, (i, j) in zip(axes_flat, pairs):
        ax.set_facecolor(background_color)
        feat_x, feat_y = features[i], features[j]

        for cluster_id in range(n_clusters):
            members = modal_label[modal_label == cluster_id].index
            sub = asset_signature.loc[asset_signature.index.intersection(members)]
            if sub.empty:
                continue
            handle = ax.scatter(
                sub[feat_x],
                sub[feat_y],
                color=colors[cluster_id],
                s=45,
                alpha=0.75,
                edgecolors="white",
                linewidth=0.4,
            )
            cluster_handles.setdefault(cluster_id, (handle, len(sub)))
            if annotate:
                dx = (lims_by_feat[feat_x][1] - lims_by_feat[feat_x][0]) * 0.008
                dy = (lims_by_feat[feat_y][1] - lims_by_feat[feat_y][0]) * 0.008
                for asset, row in sub.iterrows():
                    ax.text(
                        row[feat_x] + dx,
                        row[feat_y] + dy,
                        asset,
                        fontsize=6.5,
                        color=colors[cluster_id],
                        zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                    )

        for cluster_id, crow in centroids.iterrows():
            ax.scatter(
                crow[feat_x],
                crow[feat_y],
                color=colors[int(cluster_id)],
                s=220,
                marker="X",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )

        x_lim = lims_by_feat[feat_x]
        y_lim = lims_by_feat[feat_y]
        if not is_dcor:
            if x_lim[0] <= 0 <= x_lim[1]:
                ax.axvline(
                    0,
                    color=styles.COLORS["ref_line"],
                    linewidth=styles.lw("reference"),
                    linestyle="--",
                    alpha=0.5,
                )
            if y_lim[0] <= 0 <= y_lim[1]:
                ax.axhline(
                    0,
                    color=styles.COLORS["ref_line"],
                    linewidth=styles.lw("reference"),
                    linestyle="--",
                    alpha=0.5,
                )
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel(short_labels[i], fontsize=9)
        ax.set_ylabel(short_labels[j], fontsize=9)

    for ax in axes_flat[n_pairs:]:
        fig.delaxes(ax)

    if cluster_handles:
        ordered = sorted(cluster_handles.items())
        handles = [h for _, (h, _) in ordered]
        labels = [f"C{cid}  (n={n})" for cid, (_, n) in ordered]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=n_clusters,
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(
        f"Projeções 2D dos clusters ({method})  |  Ref: {ref.strftime('%Y-%m')}  |  "
        f"Sil. médio: {sil_mean:.3f}  |  Flip rate: {flip_rate_mean:.3f}",
        fontsize=13,
        fontweight="bold",
        y=1.06,
    )
    plt.tight_layout()
    plt.show()
