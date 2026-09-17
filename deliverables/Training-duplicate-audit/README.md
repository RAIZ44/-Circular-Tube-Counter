# Exact duplicate training-label audit

All 49 affected training images were checked against their original COCO
annotations. The 409 excess boxes are exact duplicates already present in the
exports, not boxes accidentally duplicated by preprocessing or clipping.
All 49 image hashes and reconstructed source annotations matched the manifest.

The largest two records have 203 labels but only 51 distinct boxes; many ends
were labeled four times. The three illustrated cases show repeated boxes in
orange, with their multiplicity. Green boxes occur once.

A separate versioned training manifest removes these exact repeats, preserving
the first copy, every image and every split group. It contains 1,529,155 labels
instead of 1,529,564. Round-trip identity checks passed. Original data, the frozen
split, validation labels and the reserved test are unchanged. This revision has
not been used for a training run and does not imply an accuracy improvement.

The removed labels are only 0.027% of training annotations. This repair does not
explain the full accuracy gap or justify repeated training by itself. Partial-end
conventions, missing annotations and actual model failures still need work.

See audit.json for source annotation IDs, exact repeated coordinates and hashes;
training_revision.json identifies the staged training manifest.
