"""Terminal tool: predict a prompt's dollar cost across models before sending.

Examples (run from the repo root):

    # inline prompt
    python -m app.predict "Refactor the auth module to async/await"

    # attach real file context (counts toward input tokens)
    python -m app.predict "Explain and optimize this" --file app/ledger.py --file app/models.py

    # estimate input from your real repo history (referenced files + tool + history overhead)
    python -m app.predict "Refactor the auth module" --realistic

    # set expected output size explicitly instead of the automatic estimate
    python -m app.predict "Write tests for parse()" --output-tokens 800

    # pipe a longer prompt in
    Get-Content prompt.txt | python -m app.predict --file app/api.py
"""
from __future__ import annotations

import argparse
import sys

from app.cost_predictor import (
    DEFAULT_OUTPUT_TOKENS,
    compute_input_tokens,
    detect_big_ask,
    detect_vague_prompt,
    expected_output_tokens_for,
    historical_median_output_tokens,
    predict_across_models,
    realistic_input_estimate,
)
from app.currency import (
    DEFAULT_CREDITS_PER_DOLLAR,
    DEFAULT_USD_TO_INR,
    credits_to_inr,
    credits_to_usd,
)
from app.llm_review import llm_available, review_prompt_with_llm
from app.model_pricing import recommend_model


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
                        help=f"Expected output tokens (default {DEFAULT_OUTPUT_TOKENS}); "
                             "overrides any automatic estimate.")
    parser.add_argument("--use-history", action="store_true",
                        help="Use the flat median output tokens from your real chatSessions history.")
    parser.add_argument("--realistic", action="store_true",
                        help="Estimate input from your repo's real median per-request context "
                             "(referenced files + history + tools), and estimate output from "
                             "your repo's real input->output relationship (with big-ask detection).")
    parser.add_argument("--usd", action="store_true", help="Also show cost in US dollars.")
    parser.add_argument("--inr", action="store_true", help="Also show cost in Indian rupees.")
    parser.add_argument("--credits-per-dollar", type=float, default=DEFAULT_CREDITS_PER_DOLLAR,
                        help=f"Credits per USD (default {DEFAULT_CREDITS_PER_DOLLAR:g}, "
                             "configurable -- not an official Copilot rate).")
    parser.add_argument("--usd-to-inr", type=float, default=DEFAULT_USD_TO_INR,
                        help=f"USD to INR rate (default {DEFAULT_USD_TO_INR:g}).")
    args = parser.parse_args(argv)

    prompt = _read_prompt(args.prompt)
    if not prompt:
        parser.error("No prompt provided.")

    context_files = list(args.file or [])

    extra_context = 0
    realistic_meta = None
    if args.realistic and not context_files:
        # Only fill in context from history when no explicit files were given --
        # named files are already the real context, so adding the historical
        # median on top would double-count what it's meant to represent.
        realistic_meta = realistic_input_estimate()
        extra_context = (realistic_meta or {}).get("tokens", 0) or 0

    is_big_ask, big_ask_terms = detect_big_ask(prompt)

    llm_result = review_prompt_with_llm(prompt) if llm_available() else None
    if llm_result is not None:
        is_vague, vague_reasons = llm_result["vague"], llm_result["reasons"]
        rewritten_prompt = llm_result["rewritten_prompt"]
        vague_source = "LLM-reviewed"
    else:
        is_vague, vague_reasons = detect_vague_prompt(prompt)
        rewritten_prompt = None
        vague_source = "heuristic"
    if args.output_tokens is not None:
        expected_output = args.output_tokens
        output_meta = None
    elif args.realistic:
        # Scale off what was actually TYPED, not the attached context -- how
        # much context you reference doesn't predict output size, how much
        # you're asking for does. Fall back to the full-input relationship if
        # this repo has no captured typed-text history to build that ratio from.
        typed_tokens = compute_input_tokens(prompt)["prompt_tokens"]
        full_input_tokens = compute_input_tokens(prompt, context_files, extra_context)["input_tokens"]
        output_meta = expected_output_tokens_for(typed_tokens, full_input_tokens, big_ask=is_big_ask)
        expected_output = (output_meta or {}).get("tokens") or historical_median_output_tokens() \
            or DEFAULT_OUTPUT_TOKENS
    elif args.use_history:
        expected_output = historical_median_output_tokens() or DEFAULT_OUTPUT_TOKENS
        output_meta = None
    else:
        expected_output = DEFAULT_OUTPUT_TOKENS
        output_meta = None

    result = predict_across_models(prompt, context_files, expected_output, extra_context)

    print()
    if args.realistic and extra_context:
        print(f"Realistic mode: +{extra_context:,} tokens for your median context "
              f"(scope: {realistic_meta['scope']}, {realistic_meta['sample_size']} request(s)).")
    elif args.realistic and context_files:
        print("Realistic mode: explicit --file context given, using exact file tokens "
              "(historical context estimate skipped to avoid double-counting).")
    if is_big_ask:
        print(f"Big-ask detected (matched: {', '.join(big_ask_terms)}) -- using a higher "
              f"output-token estimate for broad/open-ended work.")
    if is_vague:
        print(f"! Vague/incomplete prompt ({vague_source}): {'; '.join(vague_reasons)}. "
              f"This tends to need a clarifying follow-up turn (extra cost).")
        if rewritten_prompt:
            print(f"  Suggested rewrite: {rewritten_prompt}")
        else:
            print("  Consider naming a file, function, or expected behavior before sending.")
    if output_meta:
        print(f"Output estimate: {output_meta['tokens']:,} tokens via {output_meta['method']} "
              f"(scope: {output_meta['scope']}, {output_meta['sample_size']} request(s)).")
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
    if args.usd:
        header += f"{'TOTAL $':>10}"
    if args.inr:
        header += f"{'TOTAL Rs':>11}"
    print(header)
    print("-" * len(header))
    for m in result["models"]:
        window = m.get("context_window")
        window_str = f"{window:,}" if window else "?"
        flag = "" if m.get("fits", True) else "  ! over window"
        line = (f"{m['model']:<28}{window_str:>12}{m['input_cost']:>10.4f}"
                f"{m['output_cost']:>11.4f}{m['total_cost']:>11.4f}")
        if args.usd:
            line += f"{credits_to_usd(m['total_cost'], args.credits_per_dollar):>10.4f}"
        if args.inr:
            line += f"{credits_to_inr(m['total_cost'], args.credits_per_dollar, args.usd_to_inr):>11.2f}"
        print(line + flag)
    print("-" * len(header))

    cheapest = result["cheapest"]
    dearest = result["most_expensive"]
    if cheapest and dearest:
        spread = dearest["total_cost"] - cheapest["total_cost"]
        print(f"\nCheapest: {cheapest['model']} at {cheapest['total_cost']:.4f} credits "
              f"(vs {dearest['total_cost']:.4f} for {dearest['model']}, saves {spread:.4f} credits/request)")

    # Intent-based routing: recommend a model matched to the task, not just the
    # cheapest overall. Tier comes from the LLM's judgment when available,
    # otherwise a heuristic (broad/big-ask -> premium, else economy).
    if llm_result and llm_result.get("recommended_tier"):
        tier = llm_result["recommended_tier"]
        intent = llm_result.get("intent") or "task"
        complexity = llm_result.get("complexity") or "unknown"
        routing_reason = llm_result.get("routing_reason") or ""
        source = "LLM"
    else:
        tier = "premium" if is_big_ask else "economy"
        intent = "broad/large task" if is_big_ask else "routine task"
        complexity = "high" if is_big_ask else "low"
        routing_reason = "big-ask keywords matched" if is_big_ask else "no complexity signals"
        source = "heuristic"
    rec_model, rec_cost = recommend_model(tier, result["input_tokens"], result["expected_output_tokens"])
    extra = rec_cost - cheapest["total_cost"] if cheapest else 0.0
    premium_note = "" if extra <= 0 else f", +{extra:.4f} credits vs cheapest"
    print(f"Recommended ({source}): {rec_model} at {rec_cost:.4f} credits -- "
          f"{tier} tier for {intent} ({complexity} complexity{premium_note}).")
    if routing_reason:
        print(f"  Why: {routing_reason}")
    if args.usd or args.inr:
        print(f"Currency rates: {args.credits_per_dollar:g} credits/$ "
              f"(configurable, not an official Copilot rate)"
              + (f", {args.usd_to_inr:g} INR/$" if args.inr else "") + ".")
    print("\nRates are real Copilot credits per 1M tokens (app/model_pricing.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
