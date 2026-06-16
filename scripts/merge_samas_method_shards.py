from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if not r.get("abstained")]
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r.get("correct")) / max(n, 1),
        "abstain_rate": sum(1 for r in rows if r.get("abstained")) / max(n, 1),
        "correct_on_answered": sum(1 for r in answered if r.get("correct")) / max(len(answered), 1),
        "avg_rounds": statistics.mean(float(r.get("rounds") or 0) for r in rows) if rows else 0.0,
        "avg_elapsed_s": statistics.mean(float(r.get("elapsed_s") or 0.0) for r in rows) if rows else 0.0,
        "score_g": sum(1 for r in rows if r.get("correct")) / max(n, 1),
        "raw": rows,
    }


def load_rows(path: Path, method: str) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(method, {}).get("medqa", {}).get("raw", [])


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/merge_samas_method_shards.py <method_dir> <method>")
        return 2
    method_dir = Path(sys.argv[1])
    method = sys.argv[2]
    rows_by_id: dict[str, dict] = {}
    for row in load_rows(method_dir / "results.json", method):
        if row.get("case_id"):
            rows_by_id[row["case_id"]] = row
    for path in sorted((method_dir / "shards").glob("*.json")):
        for row in load_rows(path, method):
            if row.get("case_id"):
                rows_by_id[row["case_id"]] = row
    rows = sorted(rows_by_id.values(), key=lambda r: r.get("case_id", ""))
    out = {
        method: {
            "medqa": aggregate(rows),
            "merged_from": [str(p) for p in sorted((method_dir / "shards").glob("*.json"))],
        }
    }
    target = method_dir / "results_merged.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = out[method]["medqa"]
    print(
        f"{method}: n={metrics['n']} acc={metrics['accuracy']:.3f} "
        f"abstain={metrics['abstain_rate']:.3f} wrote={target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
