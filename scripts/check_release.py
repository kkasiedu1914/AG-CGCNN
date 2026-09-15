"""Check source syntax, notebook outputs and obvious secret patterns before publishing."""
from pathlib import Path
import ast
import json
import re

ROOT=Path(__file__).resolve().parents[1]
SECRET_PATTERNS=[
    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'AIza[0-9A-Za-z_-]{35}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def main():
    failures=[];checked=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in {'.git','.venv','runs','__pycache__','.pytest_cache'} for part in path.relative_to(ROOT).parts): continue
        if path.suffix not in {'.py','.md','.json','.ipynb','.yml','.toml','.txt','.cff'}: continue
        text=path.read_text(encoding='utf8')
        checked+=1
        if path.suffix=='.py':
            try: ast.parse(text,filename=str(path))
            except SyntaxError: failures.append(str(path.relative_to(ROOT))+': invalid Python syntax')
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            failures.append(str(path.relative_to(ROOT))+': possible credential; inspect privately')
        if path.suffix=='.ipynb':
            nb=json.loads(text)
            if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code'):
                failures.append(str(path.relative_to(ROOT))+': notebook execution output present')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Checked {checked} text artifacts: syntax, clean notebook outputs and selected credential patterns passed.')


if __name__=='__main__': main()
