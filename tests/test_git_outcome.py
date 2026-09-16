import subprocess
from pathlib import Path

import pytest

from app.git_outcome import _repo_root, git_outcome_for_case


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


pytestmark = pytest.mark.skipif(not _has_git(), reason="git not available")


@pytest.fixture
def temp_repo(tmp_path):
    _repo_root.cache_clear()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    yield repo
    _repo_root.cache_clear()


def test_returns_none_when_no_paths_in_a_repo(tmp_path):
    outside = tmp_path / "not_a_repo" / "file.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("x")
    assert git_outcome_for_case([str(outside)], since_iso=None) is None


def test_committed_file_is_kept(temp_repo):
    f = temp_repo / "kept.py"
    f.write_text("print('hello')\n")
    _git(temp_repo, "add", "kept.py")
    _git(temp_repo, "commit", "-m", "add kept")

    result = git_outcome_for_case([str(f)], since_iso=None)
    assert result is not None
    assert result["considered"] == 1
    assert result["kept"] == 1
    assert result["accepted"] is True


def test_uncommitted_file_is_not_kept(temp_repo):
    f = temp_repo / "draft.py"
    f.write_text("print('wip')\n")  # never committed

    result = git_outcome_for_case([str(f)], since_iso=None)
    assert result is not None
    assert result["considered"] == 1
    assert result["kept"] == 0
    assert result["accepted"] is False


def test_majority_rule_across_files(temp_repo):
    kept1 = temp_repo / "a.py"
    kept2 = temp_repo / "b.py"
    draft = temp_repo / "c.py"
    for f, body in [(kept1, "a"), (kept2, "b")]:
        f.write_text(body)
        _git(temp_repo, "add", f.name)
    _git(temp_repo, "commit", "-m", "add a and b")
    draft.write_text("c")  # uncommitted

    result = git_outcome_for_case([str(kept1), str(kept2), str(draft)], since_iso=None)
    assert result["considered"] == 3
    assert result["kept"] == 2
    assert result["accepted"] is True  # 2/3 >= 0.5
