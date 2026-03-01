import sys


def main() -> None:
    msg = (
        "scripts/build_audit_set.py is deprecated and disabled. "
        "Use scripts/build_dual_eval_sets.py and audit_*_paired.jsonl for strict document-level holdout."
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
