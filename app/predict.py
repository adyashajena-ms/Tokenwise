"""Terminal tool: predict a prompt's dollar cost across models before sending.

Examples (run from the repo root):

    # inline prompt
    python -m app.predict "Refactor the auth module to async/await"

    # attach real file context (counts toward input tokens)
    python -m app.predict "Explain and optimize this" --file app/ledger.py --file app/models.py

    # set expected output size, or pull the median from your real history
    python -m app.predict "Write tests for parse()" --output-tokens 800
    python -m app.predict "Write tests for parse()" --use-history

    # pipe a longer prompt in
    Get-Content prompt.txt | python -m app.predict --file app/api.py
"""
from __future__ import annotations

import argparse
import sys

from app.cost_predictor import (
    DEFAULT_OUTPUT_TOKENS,
    gather_git_context_files,
    historical_median_input_tokens,
    historical_median_output_tokens,
    predict_across_models,
)


def _read_prompt(inline: str | None) -> str:
    if inline:
        return inline
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return input("Enter your prompt: ").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.predict",
        description="Predict a prompt's cost across models before you send it.",
    )
    parser.add_argument("prompt", nargs="?", help="The prompt text (or pipe via stdin).")
    parser.add_argument("--file", "-f", action="append", metavar="PATH",
                        help="Context file to include in the input token count (repeatable).")
    parser.add_argument("--output-tokens", "-o", type=int, default=None,
                        help=f"Expected output tokens (default {DEFAULT_OUTPUT_TOKENS}).")
    parser.add_argument("--use-history", action="store_true",
                        help="Use the median output tokens from your real chatSessions history.")
    parser.add_argument("--auto", action="store_true",
                        help="Zero-config: pull context from git-changed files and size output "
                             "from your history. Just give the prompt.")
    parser.add_argument("--realistic", action="store_true",
                        help="Estimate input from your real median per-request context "
                             "(referenced files + history + tools), not just git files.")
    args = parser.parse_args(argv)

    prompt = _read_prompt(args.prompt)
    if not prompt:
        parser.error("No prompt provided.")

    context_files = list(args.file or [])
    if args.auto and not context_files:
        context_files = gather_git_context_files()

    extra_context = 0
    if args.realistic:
        extra_context = historical_median_input_tokens() or 0

    if args.output_tokens is not None:
        expected_output = args.output_tokens
    elif args.use_history or args.auto or args.realistic:
        expected_output = historical_median_output_tokens() or DEFAULT_OUTPUT_TOKENS
    else:
        expected_output = DEFAULT_OUTPUT_TOKENS

    result = predict_across_models(prompt, context_files, expected_output, extra_context)

    print()
    if args.auto:
        print(f"Auto mode: {len(context_files)} git-changed file(s) as context, "
              f"output sized from history.")
    if args.realistic and extra_context:
        print(f"Realistic mode: +{extra_context:,} tokens for your real median context "
              f"(referenced files + history + tools).")
    print(f"Prompt tokens : {result['prompt_tokens']:,}")
    if result["context_tokens"]:
        print(f"Context tokens: {result['context_tokens']:,}")
        for f in result["file_breakdown"]:
            if f.get("error"):
                print(f"  ! {f['file']} — {f['error']}")
            else:
                print(f"  + {f['file']}: {f['tokens']:,} tokens")
    print(f"Input tokens  : {result['input_tokens']:,}")
    print(f"Output tokens : {result['expected_output_tokens']:,} (assumed)")
    print()

    header = f"{'MODEL':<28}{'CONTEXT':>12}{'INPUT cr':>10}{'OUTPUT cr':>11}{'TOTAL cr':>11}"
    print(header)
    print("-" * len(header))
    for m in result["models"]:
        window = m.get("context_window")
        window_str = f"{window:,}" if window else "?"
        flag = "" if m.get("fits", True) else "  ! over window"
        print(f"{m['model']:<28}{window_str:>12}{m['input_cost']:>10.4f}"
              f"{m['output_cost']:>11.4f}{m['total_cost']:>11.4f}{flag}")
    print("-" * len(header))

    cheapest = result["cheapest"]
    dearest = result["most_expensive"]
    if cheapest and dearest:
        spread = dearest["total_cost"] - cheapest["total_cost"]
        print(f"\nCheapest: {cheapest['model']} at {cheapest['total_cost']:.4f} credits "
              f"(vs {dearest['total_cost']:.4f} for {dearest['model']}, saves {spread:.4f} credits/request)")
    print("\nRates are real Copilot credits per 1M tokens (app/model_pricing.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
