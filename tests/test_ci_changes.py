"""Editorial changes skip numerical work; package and unknown paths retain it."""

import subprocess

import pytest
from ci_changes import changed_paths, checks


@pytest.mark.parametrize(
    "paths,tests,wheel",
    [
        (["README.md", "docs/runner.md"], False, False),
        (["docs/decisions.md"], True, True),
        (["README.md", "src/toylife2/chemistry.py"], True, True),
        (["tests/test_gene_copy.py"], True, False),
        (["examples/molecular-cue.toml"], True, False),
        (["pyproject.toml", "uv.lock"], True, True),
        (["upstream/toylife"], True, True),
        ([".github/workflows/ci.yml"], True, True),
        (["unknown-file"], True, False),
        (None, True, True),
    ],
)
def test_changed_path_checks(paths, tests, wheel):
    assert checks(paths) == {"tests": tests, "wheel": wheel, "full": False}


def test_explicit_qualification_includes_editorial_and_empty_changes():
    for paths in ([], ["README.md"], None):
        assert checks(paths, full=True) == {"tests": True, "wheel": True, "full": True}


def test_missing_history_keeps_checks_enabled():
    assert changed_paths("0" * 40) is None
    assert changed_paths("--untrusted-ref") is None


def test_move_from_code_into_docs_still_runs_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    code = tmp_path / "src" / "example.py"
    code.parent.mkdir()
    code.write_text("source\n")
    git("add", ".")
    git("commit", "-qm", "initial")
    base = git("rev-parse", "HEAD")
    docs = tmp_path / "docs" / "example.md"
    docs.parent.mkdir()
    code.rename(docs)
    git("add", "-A")
    git("commit", "-qm", "move")
    paths = changed_paths(base)
    assert paths is not None and set(paths) == {"src/example.py", "docs/example.md"}
    assert checks(paths)["tests"] and checks(paths)["wheel"]
