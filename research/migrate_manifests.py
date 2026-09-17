import json
import shutil
from pathlib import Path

from pipe_counter.utils import read_jsonl, write_jsonl

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
backup = Path(r'C:\Users\Owner\Documents\Codex\2026-09-15\okay-look-though-my-folder-let\work\before-repair\legacy-manifests')
backup.mkdir(parents=True, exist_ok=True)
summary = {}
for split in ('train', 'val', 'test'):
    manifest = project / 'data' / 'processed' / f'{split}.jsonl'
    records = read_jsonl(manifest)
    missing = [r['image_path'] for r in records if not Path(r['image_path']).is_file()]
    if missing:
        raise RuntimeError(f'{split}: {len(missing)} unresolved images: {missing[:3]}')
    if not (backup / manifest.name).exists():
        shutil.copy2(manifest, backup / manifest.name)
    write_jsonl(manifest, records)
    restored = read_jsonl(manifest)
    assert [(r['sha256'], r['boxes']) for r in restored] == [(r['sha256'], r['boxes']) for r in records]
    summary[split] = {'records': len(records), 'missing_images': len(missing), 'portable': True}
print(json.dumps(summary, indent=2))
