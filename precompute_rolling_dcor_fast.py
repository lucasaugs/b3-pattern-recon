"""Precompute rolling asset-asset/asset-macro em `dcor` com a estratégia de
centragem pré-computada recomendada pelo criador do pacote `dcor`
(ver `examples/efficient_dcor.py`).

Motivação
---------
O caminho atual (`correlations.calculate_rolling_correlations`, ramo `dcor`)
chama `dcor.distance_correlation(r_col, c_col)` para *cada par ordenado* de cada
janela. Cada chamada recomputa internamente a matriz de distâncias e o
duplo-centramento das DUAS colunas. Numa matriz asset-asset (N×N) isso recomputa
a matriz centrada de cada ativo ~2N vezes por janela — O(N²) `pdist`/centragens.

Aqui centramos cada coluna UMA vez por janela (O(N) `pdist`) e combinamos os
pares com `dcor.mean_product` (em dcor 0.7; era `average_product` no exemplo
original do StackOverflow), reaproveitando a álgebra do duplo-centramento:

    R(X,Y) = sqrt( mean_product(A,B) / sqrt(mean_product(A,A) * mean_product(B,B)) )

onde A, B são as matrizes de distância duplamente centradas de X, Y. Como dcor é
simétrico, no caso asset-asset só computamos o triângulo superior e espelhamos.

Painel desbalanceado
--------------------
A implementação original faz `dropna()` PAREADO por (r_col, c_col): cada par usa
o seu próprio conjunto de linhas válidas. A matriz centrada de uma coluna só pode
ser reaproveitada entre pares se o conjunto de linhas for o mesmo. Por isso só
pré-computamos a centragem das colunas SEM NaN na janela; qualquer par que
envolva uma coluna com NaN na janela cai no fallback pairwise idêntico ao
original. Para o par (completo, completo) o `dropna` não remove nada, então o
resultado é numericamente igual ao caminho original (ver `--validate`).

O arquivo de saída é o MESMO cache canônico que o app/`correlations.py` leem
(`data/cache/corr/rolling/rolling_W{w}_dcor_{target}_A{hash}{lag}.pkl`), então
este script é um substituto direto do ramo `dcor` de
`precompute_rolling_asset_asset.py`.

Uso:
    python precompute_rolling_dcor_fast.py
    python precompute_rolling_dcor_fast.py --windows 12 24
    python precompute_rolling_dcor_fast.py --target asset-macro
    python precompute_rolling_dcor_fast.py --force
    python precompute_rolling_dcor_fast.py --validate --windows 6   # confere vs implementação atual
    python precompute_rolling_dcor_fast.py --lags 1 2 --windows 6   # caches da aba de defasagem
"""

import argparse
import os
import pickle
import time
from itertools import product

import dcor
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from correlations import (
    MACRO_FEATURES,
    ROLLING_CACHE_DIR,
    _apply_lag,
    _build_assets_hash,
    _build_lag_suffix,
    load_or_process_master_df,
)

DEFAULT_WINDOWS = [3, 6, 12, 24]


# =============================================================================
# Núcleo: dcor rolling com centragem pré-computada
# =============================================================================


def _pair_dcor(r, c, centered, self_prod, window, min_periods):
    """dcor de um par. Usa centragem pré-computada quando ambas as colunas estão
    completas na janela; caso contrário, fallback pairwise idêntico ao original."""
    if r in centered and c in centered:
        num = dcor.mean_product(centered[r], centered[c])
        denom = np.sqrt(self_prod[r] * self_prod[c])
        # Variância de distância nula (série constante na janela): dcor convenciona 0.
        if denom <= 0:
            return 0.0
        return float(np.sqrt(num / denom))

    valid_pairs = window[[r, c]].dropna()
    if len(valid_pairs) >= min_periods:
        return float(
            dcor.distance_correlation(
                valid_pairs[r].values, valid_pairs[c].values
            )
        )
    return np.nan


def compute_rolling_dcor_fast(
    master_data: pd.DataFrame,
    asset_list: list,
    window_size: int,
    target: str = "asset-asset",
    lag: int = 0,
    lag_features: list = None,
) -> pd.DataFrame:
    """Versão rápida (centragem pré-computada) do ramo `dcor` de
    `calculate_rolling_correlations`. Produz o MESMO DataFrame de saída."""
    clean_list = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]
    working_data = _apply_lag(master_data, lag, lag_features)
    target_cols = clean_list if target == "asset-asset" else list(MACRO_FEATURES)
    symmetric = target == "asset-asset"
    min_periods = max(4, int(window_size * 0.5))

    # União ordenada das colunas que precisam de matriz centrada nesta computação.
    needed_cols = list(dict.fromkeys(clean_list + target_cols))

    frames = []
    keys = []
    for i in range(window_size - 1, len(working_data)):
        window = working_data.iloc[i - window_size + 1 : i + 1]

        # 1. Centra UMA vez cada coluna sem NaN na janela (e guarda o auto-produto).
        centered = {}
        for col in needed_cols:
            vals = window[col].to_numpy()
            if not np.isnan(vals).any():
                dist = squareform(pdist(vals[:, np.newaxis]))
                centered[col] = dcor.double_centered(dist)
        self_prod = {col: dcor.mean_product(a, a) for col, a in centered.items()}

        # 2. Preenche a matriz de pares (triângulo superior + espelho se simétrico).
        local_frame = pd.DataFrame(index=clean_list, columns=target_cols, dtype=float)
        if symmetric:
            n = len(clean_list)
            for a in range(n):
                ra = clean_list[a]
                for b in range(a, n):
                    cb = clean_list[b]
                    val = _pair_dcor(ra, cb, centered, self_prod, window, min_periods)
                    local_frame.iat[a, b] = val
                    local_frame.iat[b, a] = val
        else:
            for ra in clean_list:
                for cb in target_cols:
                    local_frame.loc[ra, cb] = _pair_dcor(
                        ra, cb, centered, self_prod, window, min_periods
                    )

        frames.append(local_frame)
        keys.append(working_data.index[i])

    rolling_corr = pd.concat(frames, keys=keys, names=["TIMESTAMP", "Variável"])

    idx = pd.IndexSlice
    result = rolling_corr.loc[idx[:, clean_list], target_cols]
    # ffill POR ATIVO (ver correlations.calculate_rolling_correlations): preserva
    # a sensibilidade quando a macro fica inerte sem vazar valores entre ativos.
    return result.groupby(level=1).ffill().dropna(how="all")


# =============================================================================
# Referência lenta (ground truth) — espelha o ramo dcor original, sem cache
# =============================================================================


def reference_rolling_dcor_slow(
    master_data: pd.DataFrame,
    asset_list: list,
    window_size: int,
    target: str = "asset-asset",
    lag: int = 0,
    lag_features: list = None,
) -> pd.DataFrame:
    """Replica em memória o ramo `dcor` original de
    `calculate_rolling_correlations` (sem ler/escrever cache) para validação."""
    clean_list = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_data.columns
    ]
    working_data = _apply_lag(master_data, lag, lag_features)
    target_cols = clean_list if target == "asset-asset" else list(MACRO_FEATURES)
    min_periods = max(4, int(window_size * 0.5))

    frames = []
    keys = []
    for i in range(window_size - 1, len(working_data)):
        window = working_data.iloc[i - window_size + 1 : i + 1]
        local_frame = pd.DataFrame(index=clean_list, columns=target_cols, dtype=float)
        for r_col in clean_list:
            for c_col in target_cols:
                valid_pairs = window[[r_col, c_col]].dropna()
                if len(valid_pairs) >= min_periods:
                    local_frame.loc[r_col, c_col] = dcor.distance_correlation(
                        valid_pairs[r_col].values, valid_pairs[c_col].values
                    )
        frames.append(local_frame)
        keys.append(working_data.index[i])

    rolling_corr = pd.concat(frames, keys=keys, names=["TIMESTAMP", "Variável"])
    idx = pd.IndexSlice
    result = rolling_corr.loc[idx[:, clean_list], target_cols]
    return result.groupby(level=1).ffill().dropna(how="all")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help=f"Tamanhos de janela em meses (default: {DEFAULT_WINDOWS}).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="asset-asset",
        choices=["asset-asset", "asset-macro"],
        help="Alvo da correlação móvel (default: asset-asset).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[],
        help="Lags (meses) a precomputar por feature. Vazio = só o tensor sem lag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recomputa mesmo se o cache existir.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compara a saída rápida com a implementação original (lenta) por "
        "combinação e reporta a diferença máxima absoluta. Não escreve cache.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{int(sec):02d}s"


def _cache_path(asset_list, master_df, window, target, lag, lag_features) -> str:
    clean_list = [
        c for c in asset_list if c != "MARKET_INDEX" and c in master_df.columns
    ]
    lag_suffix = _build_lag_suffix(lag, lag_features)
    assets_hash = _build_assets_hash(clean_list)
    filename = f"rolling_W{window}_dcor_{target}_A{assets_hash}{lag_suffix}.pkl"
    return os.path.join(ROLLING_CACHE_DIR, filename)


def _validate(master_df, asset_list, window, target, lag, lag_features) -> None:
    print("  validando rápida vs original...")
    t0 = time.perf_counter()
    fast = compute_rolling_dcor_fast(
        master_df, asset_list, window, target, lag, lag_features
    )
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = reference_rolling_dcor_slow(
        master_df, asset_list, window, target, lag, lag_features
    )
    t_slow = time.perf_counter() - t0

    fast_aligned, ref_aligned = fast.align(ref, join="outer")
    diff = (fast_aligned - ref_aligned).abs()
    max_diff = float(np.nanmax(diff.to_numpy())) if diff.size else 0.0
    same_shape = fast.shape == ref.shape
    # NaN nas mesmas posições?
    nan_match = bool((fast_aligned.isna() == ref_aligned.isna()).all().all())
    speedup = t_slow / t_fast if t_fast > 0 else float("inf")

    print(f"    shape rápida={fast.shape} | original={ref.shape} | igual={same_shape}")
    print(f"    máx |Δ|={max_diff:.3e} | máscara NaN idêntica={nan_match}")
    print(
        f"    tempo rápida={format_duration(t_fast)} | "
        f"original={format_duration(t_slow)} | speedup={speedup:.1f}x"
    )
    if max_diff > 1e-9 or not nan_match or not same_shape:
        print("    >>> DIVERGÊNCIA detectada — investigar antes de confiar no cache.")
    else:
        print("    OK: saída idêntica (dentro de 1e-9).")


def main() -> None:
    args = parse_args()

    print("Carregando master_df...")
    master_df, asset_list = load_or_process_master_df()
    print(f"  {len(asset_list)} ativos, {len(master_df)} períodos mensais.")

    # Lista de features defasadas: vazia (sem lag) ou uma feature por cache.
    if args.lags:
        if args.target == "asset-macro":
            feature_list = list(MACRO_FEATURES)
        else:
            feature_list = [a for a in asset_list if a != "MARKET_INDEX"]
        lag_jobs = [([feat], lag) for lag, feat in product(args.lags, feature_list)]
    else:
        lag_jobs = [(None, 0)]

    combos = list(product(args.windows, lag_jobs))
    overall_start = time.perf_counter()
    timings = []

    print(
        f"\nPrecomputando {len(combos)} job(s) | target={args.target} | "
        f"force={args.force} | validate={args.validate}"
    )

    for i, (window, (lag_features, lag)) in enumerate(combos, start=1):
        feat_tag = f" | lag={lag}M:{lag_features[0]}" if lag_features else ""
        header = f"[{i}/{len(combos)}] W={window}M | dcor | {args.target}{feat_tag}"
        print(f"\n{header}")
        print("-" * len(header))

        if args.validate:
            _validate(master_df, asset_list, window, args.target, lag, lag_features)
            continue

        cache_path = _cache_path(
            asset_list, master_df, window, args.target, lag, lag_features
        )
        if os.path.exists(cache_path) and not args.force:
            print(f"  já em cache (use --force para recomputar): {cache_path}")
            continue

        start = time.perf_counter()
        result = compute_rolling_dcor_fast(
            master_df, asset_list, window, args.target, lag, lag_features
        )
        elapsed = time.perf_counter() - start

        os.makedirs(ROLLING_CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        n_dates = result.index.get_level_values(0).nunique()
        timings.append((window, lag, elapsed, result.shape))
        print(
            f"  OK em {format_duration(elapsed)} | {n_dates} datas | "
            f"shape={result.shape}\n  -> {cache_path}"
        )

    if not args.validate and timings:
        print(f"\n{'=' * 60}")
        print(f"Total: {format_duration(time.perf_counter() - overall_start)}")
        print("Resumo:")
        for window, lag, elapsed, shape in timings:
            lag_tag = f"lag={lag}M " if lag else ""
            print(
                f"  W={window:>2}M | {lag_tag}{format_duration(elapsed):>8} | shape={shape}"
            )


if __name__ == "__main__":
    main()
