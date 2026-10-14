"""Ephemeral CUDA qualification: modal run tests/modal_cuda.py.

Uploads code, tests and bundled model tables only. Saves complete evidence locally
before reporting failures; one T4, one invocation, no automatic retries.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("gcc", "g++", "libgomp1", "libgl1", "libx11-6")
        .pip_install("quadrants==1.3.3", "numpy==2.5.3", "pytest==9.1.1", "tomli-w==1.2.0")
        .env({"PYTHONPATH": "/workspace/src", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
        .add_local_file(ROOT / "docs/decisions.md", "/workspace/docs/decisions.md")
        .add_local_file(ROOT / "pyproject.toml", "/workspace/pyproject.toml")
        .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
        .add_local_file(ROOT / "LICENSE", "/workspace/LICENSE")
        .add_local_file(
            ROOT / "examples/replay_molecular_genome.py",
            "/workspace/examples/replay_molecular_genome.py",
        )
    )
    if modal.is_local()
    else None
)
if modal.is_local():
    for name in (
        "runtime",
        "trajectory",
        "spatial",
        "maps",
        "chemistry",
        "rollout",
        "runtime_optimisation",
        "resident_world",
        "evaluation",
        "gene_copy",
        "molecular_interface",
        "molecular_animation",
    ):
        image = image.add_local_file(
            ROOT / "tests" / f"test_{name}.py", f"/workspace/tests/test_{name}.py"
        )
    for name in ("check_cuda.py", "check_metal.py"):
        image = image.add_local_file(ROOT / "tests" / name, f"/workspace/tests/{name}")
    for name in ("data", "promoters", "prot_metabolism", "dim_metabolism"):
        image = image.add_local_dir(
            ROOT / "upstream/toylife" / name, f"/workspace/upstream/toylife/{name}"
        )

app = modal.App("toylife2-cuda-qualification")


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=900,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def qualify(test_filter: str = ""):
    import hashlib
    import io
    import json
    import os
    import subprocess
    import sys
    import zipfile

    root = Path("/workspace")
    destination = root / "qualification"
    destination.mkdir(exist_ok=True)
    receipt = {
        "test_filter": test_filter,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "source_files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src/toylife2").glob("*.py"))
        },
        "packages": json.loads(
            subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=json"], text=True
            )
        ),
    }
    os.chdir(root)
    identity = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from toylife2 import Model, ExecutionSpec
import quadrants as qd
spec = ExecutionSpec()
assert spec.backend == 'cuda', spec.backend
model = Model(execution=spec)
assert qd.lang.impl.current_cfg().arch == qd.cuda
print(json.dumps({'automatic_backend': spec.backend, 'quadrants_version': qd.__version__}))
""",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    receipt["identity"] = {
        "exit_code": identity.returncode,
        "stdout": identity.stdout,
        "stderr": identity.stderr,
    }
    code = identity.returncode
    if code == 0:
        with (destination / "cuda-tests.log").open("w") as log:
            try:
                command = [sys.executable, "tests/check_cuda.py"]
                if test_filter:
                    command += ["-k", test_filter]
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=700,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
    receipt["exit_code"] = code
    (destination / "modal-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.glob("*")):
            if path.is_file():
                archive.write(path, path.name)
    return code, output.getvalue()


@app.local_entrypoint()
def main(destination: str = "qualification/modal-cuda", test_filter: str = ""):
    import io
    import zipfile

    code, evidence = qualify.remote(test_filter)
    destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evidence.zip").write_bytes(evidence)
    with zipfile.ZipFile(io.BytesIO(evidence)) as archive:
        archive.extractall(destination)
    print(f"CUDA exit code: {code}; evidence: {destination}")
    if code:
        raise RuntimeError("CUDA qualification failed; complete returned evidence saved")
