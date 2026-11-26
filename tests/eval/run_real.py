"""Run the Atlas chat cases against the real SpeechLMM through an SSH tunnel.

Not part of the test suite and not run in CI. The model is shared with the live showcase, so the
script only talks to a loopback port, asks for an explicit flag, runs one request at a time with a
pause, and stops at the first transport error.

    ssh -N -L 19120:127.0.0.1:9120 <vm>          # in another terminal, closed afterwards
    uv run python -m tests.eval.run_real --confirm-tunnel --out /tmp/agentar-eval.jsonl

Output: one JSON line per case (question, kind, text, first token budget, wall seconds, problems),
then a summary line. Pass `--model-name` and `--protocol` if the VM profile differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from urllib.parse import urlparse

from butler.triton_client import TritonClient
from tests.eval import agentar_eval as ev


async def main(args: argparse.Namespace) -> int:
    host = urlparse(args.url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing a non loopback url: reach the model through an SSH tunnel", file=sys.stderr)
        return 2
    if not args.confirm_tunnel:
        print("pass --confirm-tunnel once the tunnel is up and the owner agreed to the load", file=sys.stderr)
        return 2
    if not ev.data_available():
        print(f"no eval data in {ev.DATA_DIR}", file=sys.stderr)
        return 2
    engine = TritonClient(
        triton_url=args.url, model_name=args.model_name, prompt_protocol=args.protocol, timeout=60
    )
    context = ev.make_context(args.chunk_chars)
    cases = ev.load_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    results = []
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            for case in cases:
                started = time.monotonic()
                result = await ev.run_case(engine, context, case)
                wall = round(time.monotonic() - started, 3)
                results.append(result)
                out.write(json.dumps({
                    "id": result.case_id, "question": case["question"], "kind": result.kind,
                    "expected_kind": result.expected_kind, "text": result.text,
                    "decided_by": result.decided_by, "prompt_chars": result.prompt_chars,
                    "wall_s": wall, "problems": result.problems,
                }, ensure_ascii=False) + "\n")
                out.flush()
                await asyncio.sleep(args.pause)
            passed = sum(r.ok for r in results)
            out.write(json.dumps({"summary": {"cases": len(results), "passed": passed}}) + "\n")
    finally:
        await engine.close()
    print(f"{passed} of {len(results)} cases pass, details in {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:19120")
    parser.add_argument("--model-name", default="speechlmm_v2_l")
    parser.add_argument("--protocol", default="general_json")
    parser.add_argument("--chunk-chars", type=int, default=600)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--only", nargs="*", help="case ids")
    parser.add_argument("--out", default="agentar-eval.jsonl")
    parser.add_argument("--confirm-tunnel", action="store_true")
    sys.exit(asyncio.run(main(parser.parse_args())))
