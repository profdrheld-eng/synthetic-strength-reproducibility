"""Exact tracked inventory, integrity and narrow secret/path checks. Read-only."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'FILES.json').read_text())['files']
    expected=set(manifest)|{'FILES.json'}
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,check=True,capture_output=True).stdout.decode().split('\0')
    tracked={name for name in tracked if name}
    if tracked!=expected:raise AssertionError({'missing':sorted(expected-tracked),'extra':sorted(tracked-expected)})
    patterns=(r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{40,}',r'AKIA[0-9A-Z]{16}',
              r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'/'+r'Users/[^\s"\']+',r'/'+r'home/[^\s"\']+')
    for name in sorted(expected):
        path=ROOT/name
        if path.is_symlink():raise AssertionError(('symlink',name))
        raw=path.read_bytes()
        if name in manifest and hashlib.sha256(raw).hexdigest()!=manifest[name]['sha256']:raise AssertionError(('digest',name))
        text=raw.decode('utf-8')
        for pattern in patterns:
            if re.search(pattern,text):raise AssertionError(('path/secret pattern',name))
    print(json.dumps({'passed':True,'tracked_files':len(expected),'bytes':sum((ROOT/n).stat().st_size for n in expected),
                      'scope':'Exact allowlist/hash/symlink/UTF-8 and selected secret/path patterns; not a proof that all possible secrets are absent'}))

if __name__=='__main__':main()
