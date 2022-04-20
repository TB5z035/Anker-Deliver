POINTS=$1
UNC_RESULT_DIR=results/$POINTS
mkdir -p $UNC_RESULT_DIR


python -m src.uncertainty.new \
    --log_dir log \
    --seed 42 \
    --test_batch_size 24 \
    --run_name unc_inference_$POINTS \
    --unc_result_dir $UNC_RESULT_DIR \
    --unc_round 50 \
    --unc_dataset ScannetVoxelization2cmtestDataset \
    --scannet_path ~/data/$POINTS/train \
    --scannet_test_path ~/data/full/train \
    --do_unc_render