for size in 128 256 512 1024 -1; do
    python sft_experiment.py --dataset_size $size --num_epochs 2
done