#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed."
  echo "Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

echo "[setup_uv] Project root: $PROJECT_ROOT"
echo "[setup_uv] Creating venv at: $VENV_DIR (python=$PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

echo "[setup_uv] Installing dependencies from $REQUIREMENTS_FILE"
uv pip install -r "$REQUIREMENTS_FILE" --python "$VENV_DIR/bin/python"

echo "[setup_uv] Verifying core imports"
"$VENV_DIR/bin/python" - <<'PY'
mods = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "numpy",
    "pandas",
]
failed = []
for m in mods:
    try:
        __import__(m)
        print(f"  [ok] {m}")
    except Exception as e:
        failed.append((m, repr(e)))

if failed:
    print("\n[setup_uv] Import verification failed:")
    for m, err in failed:
        print(f"  [fail] {m}: {err}")
    raise SystemExit(1)

print("\n[setup_uv] Environment is ready.")
PY

echo "[setup_uv] Done. Activate with: source $VENV_DIR/bin/activate"
