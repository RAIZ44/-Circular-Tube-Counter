Integration test only: one CPU training epoch on four training images, two
validation images and two test images at 128 px. This checks loading, learning,
checkpoint saving, provenance and evaluation. Its metrics do not measure model
quality. Use the original v1 model for existing prediction demonstrations, or
train a full v2 model after reviewing data quality.
