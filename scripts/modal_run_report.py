"""Export W&B metrics using the Modal secret; no local W&B key required.

uv run modal run --env cs312-nlyu scripts/modal_run_report.py --run-id RUN_ID --output-dir reports/RUN_ID
"""
import json
from pathlib import Path

import modal

app = modal.App('cs312-run-report')
image = modal.Image.debian_slim(python_version='3.11').pip_install('wandb==0.30.0')

@app.function(image=image, secrets=[modal.Secret.from_name('dl-alchemy-wandb')], timeout=300)
def fetch_report(run_path: str):
    import wandb
    run = wandb.Api().run(run_path)
    # Query dense training rows and sparse evaluation rows separately: W&B's
    # keys filter requires every requested key to be present in a row.
    history = list(run.scan_history(keys=[
        '_step', 'optimizer_step', 'train_loss', 'learning_rate', 'progress'
    ], page_size=1000))
    for key in run.summary.keys():
        if key.endswith('_loss') and key != 'train_loss':
            history.extend(run.scan_history(keys=['_step', 'optimizer_step', key], page_size=1000))
    history.sort(key=lambda row: row['_step'])
    return {'path': run_path, 'url': run.url, 'state': run.state, 'name': run.name,
            'config': dict(run.config), 'summary': dict(run.summary), 'history': history}

@app.local_entrypoint()
def main(run_id: str, output_dir: str = 'reports', entity: str = 'lyuxingjian-na', project: str = 'assignments'):
    report = fetch_report.remote(f'{entity}/{project}/{run_id}')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run-report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({'state': report['state'], 'url': report['url'], 'history_rows': len(report['history']),
                      'report': str(output / 'run-report.json')}, indent=2))
