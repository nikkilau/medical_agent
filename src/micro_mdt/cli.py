from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_case_from_text, load_cases
from .providers import MockProvider, OpenAICompatibleProvider
from .reporting import render_result, render_summary
from .visualization import render_html, render_summary_html
from .workflow import MicroMDT


def _default_cases_file() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "examples" / "cases.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Micro-MDT multi-agent workflow.")
    parser.add_argument("--case", help="Run an ad-hoc case from plain text.")
    parser.add_argument("--cases-file", type=Path, default=_default_cases_file())
    parser.add_argument("--case-id", help="Run one case from --cases-file.")
    parser.add_argument("--provider", choices=["mock", "openai-compatible"], default="mock")
    parser.add_argument("--model", help="Model name for OpenAI-compatible providers.")
    parser.add_argument("--base-url", help="Base URL for OpenAI-compatible providers.")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--interactive-human", action="store_true", help="Prompt for doctor decision when AI abstains.")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--output-html", type=Path, help="Generate HTML report(s) in the given directory.")
    parser.add_argument("--web", action="store_true", help="Start interactive web UI (http://127.0.0.1:8080).")
    parser.add_argument("--port", type=int, default=8080, help="Port for --web mode (default 8080).")
    return parser


def make_provider(args: argparse.Namespace):
    if args.provider == "mock":
        return MockProvider()
    return OpenAICompatibleProvider(base_url=args.base_url, model=args.model)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.web:
        from .webapp import run_server
        run_server(port=args.port, provider=args.provider)
        return 0

    provider = make_provider(args)
    workflow = MicroMDT(provider, max_rounds=args.max_rounds)

    if args.case:
        cases = [load_case_from_text(args.case)]
    else:
        cases = load_cases(args.cases_file)
        if args.case_id:
            cases = [case for case in cases if case.case_id == args.case_id]
            if not cases:
                parser.error(f"No case found with case_id={args.case_id}")

    results = []
    for patient_case in cases:
        result = workflow.run_case(
            patient_case,
            human_decision_callback=ask_human_decision if args.interactive_human else None,
        )
        results.append(result)
        print(render_result(result, show_trace=args.show_trace))
        print("\n" + "=" * 80 + "\n")

    if len(results) > 1:
        print(render_summary(results))

    if args.output_html:
        out_dir = args.output_html
        out_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            path = out_dir / f"{result.case.case_id}.html"
            path.write_text(render_html(result), encoding="utf-8")
            print(f"HTML report: {path}")
        if len(results) > 1:
            summary_path = out_dir / "summary.html"
            summary_path.write_text(render_summary_html(results), encoding="utf-8")
            print(f"HTML summary: {summary_path}")

    return 0


def ask_human_decision(result):
    print()
    print("=" * 60)
    print("⚠️  AI 助手提示：遇到复杂病例，AI 专家团存在未决分歧，请主任定夺！")
    print("=" * 60)
    print(f"患者概况：{result.case.title}")
    print()
    if result.human_options:
        print(result.human_options)
        print()
    print("请主任批示（输入 A/B/C 或手动输入其他意见，直接回车跳过文书生成）：")
    print("-" * 60)
    return input("主任决策> ").strip() or None


if __name__ == "__main__":
    raise SystemExit(main())

