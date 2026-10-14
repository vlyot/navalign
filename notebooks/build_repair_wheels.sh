#!/usr/bin/env bash
# Build the wheels for the private Kaggle dataset kaic87/navalign-repair-wheels (R1 repairs).
# Notebooks attached to the competition can't use the Internet, so gold_control.py's D1 installs
# from this dataset. Target: the Kaggle notebook image (Linux x86_64, CPython 3.12).
#
#   bash notebooks/build_repair_wheels.sh OUT_DIR
#   kaggle datasets create -p OUT_DIR      # first time (private by default)
#   kaggle datasets version -p OUT_DIR -m "..."   # later (run from inside OUT_DIR on Windows)
set -euo pipefail
out="${1:?usage: build_repair_wheels.sh OUT_DIR}"
mkdir -p "$out"
cd "$out"
download=(python -m pip download --quiet --only-binary=:all: --python-version 3.12 --implementation cp
          --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 -d .)

# Test packages the shipped control reported missing (plus the forum list), with their dependencies.
# D1 only installs the ones the notebook's Python doesn't already have.
"${download[@]}" inline-snapshot dirty-equals pytest-httpbin typing-inspection ujson orjson python-multipart attrs

# Every starlette release since 0.27, so each fastapi task can get the range its pyproject.toml declares.
for v in $(python -m pip index versions starlette | sed -n 's/^Available versions: //p' | tr -d ','); do
  case "$v" in 0.[0-9].*|0.1[0-9].*|0.2[0-6].*) continue ;; esac
  "${download[@]}" --no-deps "starlette==$v"
done

cat > NAVALIGN_REPAIR_WHEELS.md <<EOF
# navalign repair wheels

Wheels for the R1 test-environment repairs in navalign's notebooks/gold_control.py (D1 finds this
dataset by this file). Built by notebooks/build_repair_wheels.sh on $(date -u +%Y-%m-%d) for
Linux x86_64, CPython 3.12. Third-party packages from PyPI under their own licences.
EOF
ls -1 *.whl | wc -l
