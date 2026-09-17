"""Make a local review sheet from training/validation only; change no labels."""
import base64
import hashlib
import html
import json
from pathlib import Path
from collections import defaultdict
from pipe_counter.utils import read_jsonl

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
out = Path(__file__).resolve().parents[1] / 'outputs'
records = read_jsonl(project/'data/processed_v2/train.jsonl') + read_jsonl(project/'data/processed_v2/val.jsonl')
by_hash = {r['sha256']: r for r in records}
conflicts = json.loads((project/'data/processed_v2/annotation_conflicts.json').read_text())
eligible = [c for c in conflicts if c['sha256'] in by_hash and len(c['annotation_counts']) > 1]
eligible.sort(key=lambda c: max(c['annotation_counts'])-min(c['annotation_counts']), reverse=True)
cases = eligible[:6]
sparse = [c for c in eligible if max(c['annotation_counts']) <= 100 and c not in cases]
cases += sparse[:4]
cache = {}

def raw_annotations(path):
    source = path.parent/'_annotations.coco.json'
    if source not in cache:
        data = json.loads(source.read_text(encoding='utf-8'))
        grouped = defaultdict(list)
        for ann in data['annotations']:
            grouped[ann['image_id']].append(ann['bbox'])
        cache[source] = {i['file_name']: grouped[i['id']] for i in data['images']}
    return cache[source][path.name]

sections, manifest = [], []
for index, conflict in enumerate(cases):
    record = by_hash[conflict['sha256']]
    path = Path(record['image_path'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    image_url = 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()
    variants, seen = [], set()
    for copy in conflict['copies']:
        source_path = Path(copy)
        boxes = raw_annotations(source_path)
        fingerprint = json.dumps(sorted(boxes))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        dots = ''.join(f'<circle cx="{x+w/2}" cy="{y+h/2}" r="3"/>' for x,y,w,h in boxes)
        label = f'{source_path.parent.parent.name}: {len(boxes)} labels'
        variants.append(f'<div class="variant"><h3>{html.escape(label)}</h3><svg viewBox="0 0 {record["width"]} {record["height"]}" role="img" aria-label="Source photo with annotation centers"><image href="{image_url}" width="{record["width"]}" height="{record["height"]}"/><g class="dots">{dots}</g></svg></div>')
    key = record['sha256'][:12]
    sections.append(f'''<section data-key="{key}"><h2>Case {index+1}: {min(conflict['annotation_counts'])} to {max(conflict['annotation_counts'])} labels</h2>
<p>Current split: {record['split']}. Retained label count: {len(record['boxes'])}. Each panel shows the same photo with an export's annotation centers.</p>
<button onclick="this.parentNode.classList.toggle('hide-dots')">Show / hide label dots</button>
<div class="variants">{''.join(variants)}</div>
<label>Reviewed count (leave blank if unsure) <input class="count" type="number" min="0" step="1"></label>
<label>Notes <textarea class="notes" placeholder="Missing labels, nested pipes, duplicate labels, uncertain visibility..."></textarea></label></section>''')
    manifest.append({'key':key, 'sha256':record['sha256'], 'split':record['split'],
                     'image':record['image_path'], 'export_counts':conflict['annotation_counts'], 'retained_count':len(record['boxes'])})

nested = next(r for r in records if r['source_file_name'] == 'pp1_jpg.rf.a3bb7e4a63273bd342e030afa5463505.jpg')
url = 'data:image/jpeg;base64,'+base64.b64encode(Path(nested['image_path']).read_bytes()).decode()
intro = f'''<h1>Pipe counting: label decisions</h1>
<p>These examples expose disagreements between existing exports. They are not verified ground truth. Only training and validation images appear; the reserved test set is excluded. This sheet changes no labels.</p>
<section><h2>First: what counts when pipes are nested?</h2><p>The existing annotation below contains {len(nested['boxes'])} labels. Smaller visible pipes sit inside larger pipes, so the intended rule changes the answer.</p>
<img class="nested" src="{url}" alt="Nested pipe ends requiring a counting rule">
<label>Intended rule <select id="rule"><option value="unresolved">Not decided</option><option>Every visible pipe end, including nested pipes</option><option>Only outer pipes</option></select></label></section>
<p>The following {len(cases)} examples are selected for review, not a representative accuracy sample. Other disagreements remain in the dataset. Green dots show annotation centers.</p>'''
document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pipe label review</title><style>
body{font:17px/1.5 system-ui;background:#f3f5f7;color:#14202e;max-width:1400px;margin:auto;padding:24px}h1{font-size:34px}h2{font-size:24px}h3{font-size:17px}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}section{background:white;border:1px solid #d7dee6;border-radius:12px;padding:24px;margin:24px 0}svg{width:100%;background:#e7eaf0}.dots{fill:#23ff69;stroke:#152017;stroke-width:.6}.hide-dots .dots{display:none}button{background:#123e63;color:white;border:0;border-radius:6px;padding:12px 18px;cursor:pointer;font-size:16px}label{display:block;margin:20px 0 0}input,textarea,select{display:block;font:inherit;padding:8px;border:1px solid #8795a6;border-radius:5px}textarea{width:95%;height:70px}.nested{width:min(100%,640px)}footer{position:sticky;bottom:0;background:#f3f5f7ef;padding:16px;border-top:1px solid #bbb}
</style><body>'''+intro+''.join(sections)+'''<footer><button onclick="saveReview()">Download review notes</button> Notes stay in this page until downloaded. Downloading does not alter the dataset.</footer>
<script>function saveReview(){const result={status:'user_review_notes_not_applied',nested_rule:document.getElementById('rule').value,reviewed_at:new Date().toISOString(),cases:[...document.querySelectorAll('section[data-key]')].map(s=>({key:s.dataset.key,count:s.querySelector('.count').value===''?null:Number(s.querySelector('.count').value),notes:s.querySelector('.notes').value}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='pipe-label-review-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}</script></body></html>'''
(out/'Pipe-label-review.html').write_text(document, encoding='utf-8')
(out/'Pipe-label-review-index.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'review_cases':len(cases),'eligible_development_conflicts':len(eligible),'counts':[[m['split'],m['export_counts']] for m in manifest],'file':str(out/'Pipe-label-review.html')}))
