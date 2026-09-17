import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from pipe_counter.splitting import split_overlap_report
from pipe_counter.utils import read_jsonl, write_jsonl

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
splits = {s: read_jsonl(project/'data'/'processed_v2'/f'{s}.jsonl') for s in ('train','val','test')}
overlap = split_overlap_report(splits)
assert not any(v for pairs in overlap.values() for v in pairs.values())
missing = [r['image_path'] for records in splits.values() for r in records if not Path(r['image_path']).is_file()]
assert not missing, missing[:3]
groups = Counter(r['split_group'] for records in splits.values() for r in records)
summary = {'images': sum(map(len,splits.values())), 'splits': {s:len(rs) for s,rs in splits.items()},
           'groups':len(groups), 'largest_group':max(groups.values()), 'missing_images':len(missing), 'overlap':overlap}
output = Path(r'C:\Users\Owner\Documents\Codex\2026-09-15\okay-look-though-my-folder-let\outputs')
output.mkdir(exist_ok=True)
(output/'verification.json').write_text(json.dumps(summary,indent=2))
for s, records in splits.items():
    # A tiny, explicitly separate integration check; these are not accuracy results.
    selected = sorted((r for r in records if 3 <= len(r['boxes']) <= 30), key=lambda r:r['sha256'])[:4 if s=='train' else 2]
    write_jsonl(project/'data'/'smoke_v2'/f'{s}.jsonl',selected)
versions = sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()
                   if d.metadata['Name'].lower().replace('_','-') not in {'pipe-center-counter','pip'}})
(project/'requirements-local-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/xpu\n'+'\n'.join(versions)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
