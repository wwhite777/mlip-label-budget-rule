"""Local version-control metadata must not change the distributed file scan."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from check_public import RULES, scan


ROOT = Path(__file__).resolve().parents[1]


def planted_content():
    return '\n'.join(sample for label, _, sample in RULES
                     if label in {'personal path', 'email address'})


def test_scan_in_copy_with_local_git_metadata(tmp_path):
    root = tmp_path / 'repo'
    files = json.loads((ROOT / 'PUBLIC_FILES.json').read_text())['files']
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    for name in ['config', 'logs/HEAD', 'objects/ab/012345']:
        path = root / '.git' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\x00' + planted_content().encode())

    command = [sys.executable, 'tools/check_public.py', '--all-files']
    result = subprocess.run(command + ['--self-test'], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['pass_all'] and report['skipped_files'] == 3
    assert report['files'] == len(json.loads((root / 'PUBLIC_FILES.json').read_text())['files'])
    assert len(report['planted_checks']) == len(RULES)
    assert all(check['planted_file_rejected'] for check in report['planted_checks'])

    # The same content remains prohibited in a file from the public inventory.
    with (root / 'README.md').open('a') as stream:
        stream.write('\n' + planted_content() + '\n')
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['skipped_files'] == 3
    assert any(f['file'] == 'README.md' and f['rule'] == 'personal path'
               for f in report['failures'])
    assert any(f['file'] == 'README.md' and f['rule'] == 'email address'
               for f in report['failures'])


@pytest.mark.parametrize('name', ['.git', '.hg', '.svn', '.bzr', '_darcs', 'CVS'])
def test_metadata_directories_at_any_depth(tmp_path, name):
    for parent in [tmp_path, tmp_path / 'nested' / 'deeper']:
        path = parent / name / 'private.txt'
        path.parent.mkdir(parents=True)
        path.write_text(planted_content())
    report = scan(tmp_path)
    assert report['pass_all'] and report['files'] == 0 and report['skipped_files'] == 2


@pytest.mark.parametrize('name', ['.git', '.hg', '.svn', '.bzr', '_darcs', 'CVS', '.git-extra'])
def test_ordinary_files_are_not_metadata_directories(tmp_path, name):
    (tmp_path / name).write_text(planted_content())
    report = scan(tmp_path)
    assert not report['pass_all'] and report['files'] == 1 and report['skipped_files'] == 0


def test_similarly_named_directory_is_scanned(tmp_path):
    path = tmp_path / '.git-extra' / 'private.txt'
    path.parent.mkdir()
    path.write_text(planted_content())
    report = scan(tmp_path)
    assert not report['pass_all'] and report['files'] == 1 and report['skipped_files'] == 0
