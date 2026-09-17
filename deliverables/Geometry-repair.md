# Image and annotation alignment repair

The detector's first full run stopped because some image dimensions differed
from their annotation metadata. No completed detector epoch was lost.

All 11,982 training and validation images were checked. Twenty were affected:
17 training and 3 validation. Twelve contained labels; their box coordinates
needed a 90-degree rotation to align with the stored photo. Eight had no labels.
All twelve labeled cases were visually checked using alternative overlays.

The repair uses an explicit per-file correction list and verifies each image's
fingerprint. The original photos, label files, count labels, and split membership
remain unchanged. Unknown mismatches still produce an error for review.

Example sheets in Geometry-review show direct, clockwise, counterclockwise, and
scaled overlays, left to right. The third panel is the confirmed alignment.
This is an AI spatial-alignment review, not verification that every visible pipe
is annotated. Existing annotation completeness issues remain.

Twenty-six regression tests pass. The original best model remains at 28.15%
exact validation counts; the detector has no full validation score yet.
The reserved test set was not inspected in this audit.
