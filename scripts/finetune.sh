python -m src.uncertainty.train \
    --seed 42 \
    --model Res16UNet34C \
    --train_batch_size 1 \
    --max_iter 30 \
    --train_dataset InferenceDataset \
    --num_workers 1