#!/usr/bin/env bash
# Build the conda env for the Angelozzi & Arcolezi (PETS 2026) DP+fairness
# intervention benchmark. Kept entirely separate from our `sdg` env: this one
# pulls aif360 + inFairness + smartnoise, which conflict with our pinned stack.
set -euo pipefail
CONDA_BASE=/home/golobs/miniconda3
ENV=dpfair
BENCH=/home/golobs/CausalFairnessInSDG/baselines/dp-fair-intervention-benchmark

source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV}\s"; then
  conda create -y -n "$ENV" python=3.10
fi
conda activate "$ENV"

PIP="$CONDA_BASE/envs/$ENV/bin/pip"

# CPU-only torch first, so inFairness does not drag in the CUDA wheels (~2.5GB)
"$PIP" install --no-input torch --index-url https://download.pytorch.org/whl/cpu

# The benchmark itself, from source, so we get data/ and example/ paths.
"$PIP" install --no-input -e "$BENCH"

# private-pgm: the same library our MST uses, at the revision the benchmark pins.
"$PIP" install --no-input \
  "git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca"

echo "=== SMOKE ==="
"$CONDA_BASE/envs/$ENV/bin/python" - <<'PY'
import aif360, fairlearn, xgboost, torch, sklearn, pandas
from BenchmarkDPFair.Benchmark import benchmark, BenchmarkInfo, BenchmarkDatasetConfig
from aif360.algorithms.preprocessing import Reweighing, LFR, DisparateImpactRemover
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction, GridSearchReduction
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
print("aif360", aif360.__version__ if hasattr(aif360,"__version__") else "?")
print("torch", torch.__version__, "sklearn", sklearn.__version__, "pandas", pandas.__version__)
print("IMPORTS OK")
PY
echo "=== DONE ==="
