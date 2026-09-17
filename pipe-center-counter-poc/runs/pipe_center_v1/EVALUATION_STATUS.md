This is the original trained model. Its weights and saved metrics are unchanged.
The original train/validation/test split has related source-image overlap.
The saved 17.7% exact-count test accuracy is a historical result, not an independent
field benchmark. Merely evaluating these weights on the new v2 split does not
remove prior exposure to training images.

For field benchmark overlap checks, supply both data/processed/train.jsonl and
data/processed/val.jsonl. Train a fresh model on processed_v2 before reporting
results for its test split. Capture separate field photos for deployment assessment.
