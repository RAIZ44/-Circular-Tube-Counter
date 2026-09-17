import json
from pathlib import Path
from datetime import datetime,timezone
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
audit=json.loads((project/'runs/supervision/image_dimensions_audit.json').read_text())
images={}
for entry in audit['mismatches']:
    assert entry['decoded']==entry['ignore_orientation']==entry['expected'][::-1]
    images[entry['sha256']]={'annotation_size':entry['expected'],'decoded_size':entry['decoded'],
        'label_count':entry['labels'],'operation':'labels_ccw90' if entry['labels'] else 'empty_labels_metadata_only',
        'split':entry['split'],'image':entry['image'],
        'review':'AI visual comparison of direct, CW, CCW, and scaled box overlays; CCW alignment verified for every labeled mismatch. Empty records preserve zero labels.',
        'annotation_counts_unchanged':True}
report={'version':1,'created_at':datetime.now(timezone.utc).isoformat(),'test_accessed':False,
    'purpose':'Explicit per-image geometry adapter. Original images, frozen manifests, grouping and count labels remain unchanged.',
    'review_scope':'Spatial alignment only, not approval of annotation completeness or semantic counting rules.',
    'images':images}
(project/'data/geometry_overrides_v1.json').write_text(json.dumps(report,indent=2))
print({'images':len(images),'labeled_images':sum(bool(e['label_count']) for e in images.values()),'train':sum(e['split']=='train' for e in images.values()),'val':sum(e['split']=='val' for e in images.values())})
