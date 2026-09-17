# Dataset contract

Train, eval, and predict **do not depend on the synthetic generator**. They only
read this layout and schema. Swap in real photos by writing the same files.

## Layout

```
data/
  data.yaml              # index (YOLO-style)
  manifest.json          # per-image counts (optional; eval uses labels)
  splits.json            # split summary (optional)
  images/
    train/*.png
    val/*.png
    test/*.png
  labels/
    train/*.txt          # same stem as the image
    val/*.txt
    test/*.txt
```

Supported image suffixes: `.png`, `.jpg`, `.jpeg`.

## Label schema (YOLO)

Each `labels/<split>/<stem>.txt` has zero or more lines:

```
<class_id> <xc> <yc> <w> <h>
```

| Field | Meaning |
| --- | --- |
| `class_id` | Always `0` (`tube`) |
| `xc`, `yc` | Box center, normalized to `[0, 1]` by image width / height |
| `w`, `h` | Box width / height, normalized to `[0, 1]` |

The box is the **axis-aligned bounding square of the tube's outer circle**.
Count = number of lines in the label file. An empty file is a valid 0-tube image.

Example with two tubes:

```
0 0.512000 0.441000 0.140000 0.140000
0 0.210000 0.730000 0.090000 0.090000
```

## `data.yaml`

Written by the generator so the split folders and class name are explicit:

```yaml
path: data
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: tube
label_format: yolo
```

## Using real images

1. Place images under `data/images/{train,val,test}/`.
2. Write matching YOLO label files under `data/labels/{train,val,test}/`.
3. Keep `configs/default.yaml` `generate.image_size` equal to the training resize
   (non-square images are bilinear-resized; boxes are scaled with the image).
4. Run train / evaluate as usual. You do not need `python -m tube_count.generate`.
