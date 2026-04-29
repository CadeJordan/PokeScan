"""Run PSA collect across multiple tokens, each scanning a separate range.

Each PSA API token has its own 100-call daily quota, so this fan-out
runs the harvester once per token (in series) with a different start
cert per token, then prints a unified DB summary at the end.

Usage:
    python -m ml.data.collect_all \\
        --starts 76000090,78000080,80000080,82000080,85000080,88000080 \\
        --count 200

`--count` is intentionally generous; the per-token `--daily-limit` will
cut each scan off cleanly when its quota is exhausted.
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


async def _run_all(
    starts: list[int],
    token_indices: list[int],
    count: int,
    concurrency: int,
    daily_limit: int,
    abort_after_misses: int,
) -> None:
    settings = get_settings()
    available = set(settings.psa_api_tokens)
    plan = dict(zip(token_indices, starts, strict=True))
    print(f"\navailable tokens: {sorted(available)}")
    print(f"plan            : {plan}")
    print(f"per-token       : count={count}, daily_limit={daily_limit}, concurrency={concurrency}\n")

    for idx, start in plan.items():
        if idx not in available:
            print(f"--- skip token {idx} (not configured) ---")
            continue
        print(f"\n=== token {idx}: {start:,} -> +{count} ===")
        try:
            await _run(
                start_cert=start,
                count=count,
                concurrency=concurrency,
                daily_limit=daily_limit,
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
        _run_all(start_list, token_list, count, concurrency, daily_limit, abort_after_misses)
    )


if __name__ == "__main__":
    app()
