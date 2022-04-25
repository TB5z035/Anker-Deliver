python -m src.uncertainty.inference \
    --seed 42 \
    --model Res16UNet34C \
    --val_dataset InferenceDataset \
    --val_batch_size 1 \
    --num_workers 4