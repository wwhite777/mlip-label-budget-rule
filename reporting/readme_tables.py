"""Generate exact-value release tables from the small reported result files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('CPU-P', 'CPU-S', 'FT-P', 'FT-S', 'CPU-R')
BEGIN = '<!-- BEGIN GENERATED TABLES -->'
END = '<!-- END GENERATED TABLES -->'


def render():
    lines = ['Values below are copied as round-trip decimal values from the released readings; no new fitting or inference is performed.', '',
             '| Stage | Design | Family | Contrast | AP increment | Reading |',
             '|---|---|---|---|---:|---|']
    sources = {}
    for stage in STAGES:
        p = ROOT / 'results' / stage / 'readings.json'
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows = json.loads(p.read_text())
        for row in rows:
            if row['design'] != 'P':
                continue
            if row['family'] not in ('F1', 'F2', 'F4', 'F5', 'F6'):
                continue
            lines.append('| ' + ' | '.join(map(str, (stage, row['design'], row['family'], row['contrast'],
                                                      row.get('estimate'), row['reading']))) + ' |')
    lines += ['', 'Secondary readings are in the corresponding `results/CPU-S`, `results/FT-S` and `results/CPU-R/readings.json` files. '
              'Their designs and family adjustments remain separate from the primary rules.', '',
              '| Rule | Status | Low-budget requirements met | Qualifying high-budget contrasts |',
              '|---|---|---|---|']
    for stage, label in (('CPU-P', 'v2.3.1'), ('CPU-R', 'v2.4 extension')):
        p = ROOT / 'results' / stage / 'judge.json'
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        row = json.loads(p.read_text())
        low = row.get('low_budgets', row.get('low_budget_conditions', {}))
        high = row.get('qualifying_high_contrasts', row.get('high_budget_witnesses', []))
        lines.append(f'| {label} | {row["status"]} | {json.dumps(low, sort_keys=True)} | {json.dumps(high)} |')
    return '\n'.join(lines) + '\n', sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    table, sources = render()
    readme = ROOT / 'README.md'
    text = readme.read_text()
    before, rest = text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    expected = before + BEGIN + '\n\n' + table + '\n' + END + after
    provenance = json.dumps(dict(sources=sources, table_sha256=hashlib.sha256(table.encode()).hexdigest()), indent=2) + '\n'
    target = ROOT / 'results/readme_tables_provenance.json'
    if a.check:
        assert text == expected, 'README table differs from result files'
        assert target.read_text() == provenance, 'table provenance drift'
    else:
        readme.write_text(expected)
        target.write_text(provenance)
    print(json.dumps(dict(pass_all=True, check_only=a.check, sources=len(sources))))


if __name__ == '__main__':
    main()
