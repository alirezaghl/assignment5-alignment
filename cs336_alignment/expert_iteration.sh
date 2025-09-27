TRAIN_PATH="/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/train.jsonl"
TEST_PATH="/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/test.jsonl"
SCRIPT_PATH="/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/expert_iteration.py"

mkdir -p ./ei_results

# batch sizes -> 512, 1024
python $SCRIPT_PATH --train_path $TRAIN_PATH --test_path $TEST_PATH --n_ei_steps 5 --n_ei_batch_sizes 512 --n_ei_rollouts 4 --n_epochs_sft 2 --filtered_data_path "./ei_results/db512.json"

python $SCRIPT_PATH --train_path $TRAIN_PATH --test_path $TEST_PATH --n_ei_steps 5 --n_ei_batch_sizes 1024 --n_ei_rollouts 4 --n_epochs_sft 2 --filtered_data_path "./ei_results/db1024.json"

# rollouts -> 2, 8

python $SCRIPT_PATH --train_path $TRAIN_PATH --test_path $TEST_PATH --n_ei_steps 5 --n_ei_batch_sizes 1024 --n_ei_rollouts 2 --n_epochs_sft 2 --filtered_data_path "./ei_results/g2.json"

python $SCRIPT_PATH --train_path $TRAIN_PATH --test_path $TEST_PATH --n_ei_steps 5 --n_ei_batch_sizes 1024 --n_ei_rollouts 8 --n_epochs_sft 2 --filtered_data_path "./ei_results/g8.json"
