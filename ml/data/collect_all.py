"""Run PSA collect across multiple tokens, each scanning a separate range.

Each PSA API token has its own daily quota (often 100 calls on the free
tier; paid keys can be higher). This fan-out runs the harvester once per
token (in series) with a different start cert per token, then prints a
unified DB summary at the end.

Usage:
    python -m ml.data.collect_all \\
        --starts 76000090,78000080,80000080,82000080,85000080,88000080 \\
        --count 200 --daily-limit-token0 2000

`--count` is intentionally generous; per-token `--daily-limit` / optional
`--daily-limit-token0` cut each scan off when that token's quota is
exhausted.
"""

from __future__ import annotations

import asyncio

import typer

from backend.app.core.config import get_settings
from ml.data.collect import _run
from ml.data.db import connect, grade_distribution, pokemon_cert_count

app = typer.Typer(add_completion=False, help="Run PSA collect across multiple tokens at once.")


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _effective_token0_count(
    count: int,
    daily_limit_token0: int | None,
    count_token0: int | None,
) -> int:
    """Enough cert attempts for token 0 to possibly use a high daily_limit.

    Pokemon hits use ~2 API calls each (details + images); allow up to
    ``2 * daily_limit_token0`` attempts when only the higher quota is set.
    """
    if count_token0 is not None:
        return count_token0
    if daily_limit_token0 is None:
        return count
    return max(count, min(8000, daily_limit_token0 * 2))


async def _run_all(
    starts: list[int],
    token_indices: list[int],
    count: int,
    concurrency: int,
    daily_limit: int,
    daily_limit_token0: int | None,
    count_token0: int | None,
    abort_after_misses: int,
) -> None:
    settings = get_settings()
    available = set(settings.psa_api_tokens)
    plan = dict(zip(token_indices, starts, strict=True))
    t0_cnt = _effective_token0_count(count, daily_limit_token0, count_token0)
    print(f"\navailable tokens: {sorted(available)}")
    print(f"plan            : {plan}")
    print(
        f"defaults        : count={count}, daily_limit={daily_limit}, concurrency={concurrency}"
    )
    if daily_limit_token0 is not None:
        print(
            f"token 0 override: daily_limit={daily_limit_token0}, "
            f"max cert attempts={t0_cnt}"
        )
    print()

    for idx, start in plan.items():
        if idx not in available:
            print(f"--- skip token {idx} (not configured) ---")
            continue
        lim = daily_limit_token0 if idx == 0 and daily_limit_token0 is not None else daily_limit
        n_try = t0_cnt if idx == 0 and daily_limit_token0 is not None else count
        print(f"\n=== token {idx}: {start:,} -> +{n_try} (daily_limit={lim}) ===")
        try:
            await _run(
                start_cert=start,
                count=n_try,
                concurrency=concurrency,
                daily_limit=lim,
                abort_after_misses=abort_after_misses,
                token_index=idx,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"!!! token {idx} failed: {exc!r}")

    print("\n\n========== FINAL DB STATE ==========")
    with connect(settings.db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM certs").fetchone()[0]
        pokemon = conn.execute("SELECT COUNT(*) FROM certs WHERE is_pokemon=1").fetchone()[0]
        with_imgs = pokemon_cert_count(conn)
        print(f"total certs    : {total}")
        print(f"pokemon rows   : {pokemon}")
        print(f"pokemon w/imgs : {with_imgs}")
        print(f"grade hist     : {grade_distribution(conn)}")


@app.command()
def main(
    starts: str = typer.Option(
        ...,
        help="Comma-separated start_cert per token, e.g. '76000090,78000080,...'.",
    ),
    count: int = typer.Option(
        200,
        help="Max certs to attempt per token; --daily-limit will stop sooner.",
    ),
    concurrency: int = typer.Option(4, help="Concurrent in-flight requests per token."),
    daily_limit: int = typer.Option(100, help="Per-token daily PSA call cap."),
    daily_limit_token0: int | None = typer.Option(
        None,
        "--daily-limit-token0",
        help="Override daily cap for PSA_API_TOKEN0 only (e.g. 2000 on a paid key).",
    ),
    count_token0: int | None = typer.Option(
        None,
        "--count-token0",
        help="Max cert attempts for token 0 (default: scales with --daily-limit-token0).",
    ),
    abort_after_misses: int = typer.Option(
        25,
        help="Abort early if this many consecutive 404s with zero hits.",
    ),
    tokens: str = typer.Option(
        "0,1,2,3,4,5",
        help="Comma-separated token indices, paired positionally with --starts.",
    ),
) -> None:
    start_list = _parse_int_list(starts)
    token_list = _parse_int_list(tokens)
    if len(start_list) != len(token_list):
        raise typer.BadParameter(
            f"--starts has {len(start_list)} entries but --tokens has {len(token_list)}"
        )
    asyncio.run(
        _run_all(
            start_list,
            token_list,
            count,
            concurrency,
            daily_limit,
            daily_limit_token0,
            count_token0,
            abort_after_misses,
        )
    )


if __name__ == "__main__":
    app()
