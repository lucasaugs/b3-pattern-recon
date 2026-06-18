"""Precomputa o tensor rolling asset-asset para todas as combinações (window, method).

Cada combinação é cacheada em `data/cache/corr/rolling/` por
`calculate_rolling_correlations`. Rodar sem `--force` é seguro: combinações já
cacheadas são puladas (a função detecta o arquivo e retorna do disco).

Com `--lags`, precomputa também os tensores defasados que a aba "Análise de lag"
consome. O cache de lag é por *ativo-feature* (`lag_features=[ativo]`, igual à
chamada do plot), então cada (ativo, lag, window, method) vira um arquivo: um
tensor por ativo-feature serve todos os alvos fatiados contra ele. Isso é
O(N_ativos) chamadas por (lag, window, method) — pesado em `dcor`.

Uso:
    python precompute_rolling_asset_asset.py
    python precompute_rolling_asset_asset.py --methods spearman
    python precompute_rolling_asset_asset.py --windows 12 24 --methods dcor
    python precompute_rolling_asset_asset.py --force
    # Lag 1 e 2 meses para cada ativo (janela/método da aba de lag):
    python precompute_rolling_asset_asset.py --lags 1 2 --windows 6 --methods spearman
    # Lag 1 e 2 meses asset-macro (feature = cada IMec):
    python precompute_rolling_asset_asset.py --target asset-macro --lags 1 2 --windows 6 --methods spearman
"""

import argparse
import time
from itertools import product

from correlations import (
    MACRO_FEATURES,
    calculate_rolling_correlations,
    load_or_process_master_df,
)

DEFAULT_WINDOWS = [3, 6, 12, 24]
DEFAULT_METHODS = ["spearman", "dcor"]


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
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        choices=["spearman", "pearson", "dcor"],
        help=f"Métodos de correlação (default: {DEFAULT_METHODS}).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="asset-asset",
        choices=["asset-asset", "asset-macro"],
        help="Alvo da correlação móvel (default: asset-asset). Com --lags, define "
        "a feature defasada: ativos (asset-asset) ou IMecs (asset-macro).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[],
        help="Lags (meses) a precomputar por feature. Vazio = só o tensor "
        "sem lag. Ex.: --lags 1 2 gera os caches da aba de defasagem.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recomputa mesmo se o cache existir.",
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


def main() -> None:
    args = parse_args()

    print("Carregando master_df...")
    master_df, asset_list = load_or_process_master_df()
    print(f"  {len(asset_list)} ativos, {len(master_df)} períodos mensais.")

    overall_start = time.perf_counter()
    timings = []

    if args.lags:
        # Modo defasado: um cache por (feature, lag, window, method). A feature é
        # cada ativo (asset-asset) ou cada IMec (asset-macro), conforme --target.
        if args.target == "asset-macro":
            feature_list = list(MACRO_FEATURES)
            feature_kind = "IMec"
        else:
            feature_list = [a for a in asset_list if a != "MARKET_INDEX"]
            feature_kind = "ativo"
        combos = list(product(args.windows, args.methods, args.lags))
        total_calls = len(combos) * len(feature_list)
        print(
            f"\nPrecomputando lag ({args.target}) para {len(feature_list)} "
            f"{feature_kind}(s)-feature × {len(combos)} (window, method, lag) = "
            f"{total_calls} chamadas (force={args.force}):"
        )
        for w, m, lag in combos:
            print(f"  - W={w:>2}M, method={m}, lag={lag}M")

        call = 0
        for window, method, lag in combos:
            header = f"W={window}M | method={method} | lag={lag}M"
            print(f"\n{header}")
            print("-" * len(header))
            start = time.perf_counter()
            for feature in feature_list:
                call += 1
                calculate_rolling_correlations(
                    master_data=master_df,
                    asset_list=asset_list,
                    window_size=window,
                    method=method,
                    target=args.target,
                    lag=lag,
                    lag_features=[feature],
                    force=args.force,
                )
                print(f"  [{call}/{total_calls}] {feature}")
            elapsed = time.perf_counter() - start
            timings.append((window, method, lag, elapsed))
            print(f"  OK em {format_duration(elapsed)}")

        print(f"\n{'=' * 60}")
        print(f"Total: {format_duration(time.perf_counter() - overall_start)}")
        print("Resumo:")
        for window, method, lag, elapsed in timings:
            print(
                f"  W={window:>2}M | {method:<8} | lag={lag}M | "
                f"{format_duration(elapsed):>8}"
            )
        return

    combos = list(product(args.windows, args.methods))
    print(f"\nPrecomputando {len(combos)} combinação(ões) (force={args.force}):")
    for w, m in combos:
        print(f"  - W={w:>2}M, method={m}")

    for i, (window, method) in enumerate(combos, start=1):
        header = f"[{i}/{len(combos)}] W={window}M | method={method}"
        print(f"\n{header}")
        print("-" * len(header))

        start = time.perf_counter()
        result = calculate_rolling_correlations(
            master_data=master_df,
            asset_list=asset_list,
            window_size=window,
            method=method,
            target=args.target,
            force=args.force,
        )
        elapsed = time.perf_counter() - start
        timings.append((window, method, elapsed, result.shape))

        n_dates = result.index.get_level_values(0).nunique()
        print(
            f"  OK em {format_duration(elapsed)} | "
            f"{n_dates} datas | shape={result.shape}"
        )

    total = time.perf_counter() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Total: {format_duration(total)}")
    print("Resumo:")
    for window, method, elapsed, shape in timings:
        print(
            f"  W={window:>2}M | {method:<8} | {format_duration(elapsed):>8} | shape={shape}"
        )


if __name__ == "__main__":
    main()
