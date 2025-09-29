import torch
import numpy as np
from typing import Literal
from vllm import LLM, SamplingParams
from sft_helper import load_dataset
from unittest.mock import patch
import random
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse
import os
from drgrpo_grader import r1_zero_reward_fn
from grpo_helper import compute_group_normalized_rewards, compute_policy_gradient_loss, grpo_microbatch_train_step
from sft_helper import tokenize_prompt_and_output, get_response_log_probs
from utils import load_dataset, dataset_f, init_vllm, load_policy_into_vllm_instance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='grpo_gsm8k')
    parser.add_argument('--train_dataset_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/train.jsonl')
    parser.add_argument('--test_dataset_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/test.jsonl')
    parser.add_argument('--n_grpo_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--advantage_eps', type=float, default=1e-6)
    parser.add_argument('--rollout_batch_size', type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument('--epochs_per_rollout_batch', type=int, default=1) 
    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--sampling_temperature', type=float, default=1.0)
    parser.add_argument('--sampling_min_tokens', type=int, default=4)
    parser.add_argument('--sampling_max_tokens', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"], default="reinforce_with_baseline")
    parser.add_argument('--use_std_normalization', action='store_true', default=True)
    parser.add_argument('--use_length_normalization', action='store_true', default=False)
    parser.add_argument('--prompt_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt')
    parser.add_argument('--output_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/grpo_results.json')
    parser.add_argument('--model_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/models/Qwen2.5-Math-1.5B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

args = get_args()
train_ds_path, test_ds_path = args.train_dataset_path, args.test_dataset_path
prompt_path, output_path = args.prompt_path, args.output_path
gradient_accumulation_steps = args.gradient_accumulation_steps
lr = args.learning_rate
gradient_clipping = args.gradient_clipping
n_grpo_steps = args.n_grpo_steps
advantage_eps = args.advantage_eps
clip_range = args.clip_range
rollout_batch_size = args.rollout_batch_size
train_batch_size = args.train_batch_size
epochs_per_rollout_batch = args.epochs_per_rollout_batch
group_size = args.group_size
sampling_temperature = args.sampling_temperature
sampling_min_tokens = args.sampling_min_tokens
sampling_max_tokens = args.sampling_max_tokens
gpu_memory_utilization = args.gpu_memory_utilization
loss_type = args.loss_type
use_std_normalization = args.use_std_normalization
use_length_normalization = args.use_length_normalization
seed = args.seed
device = args.device
model_path = args.model_path


torch.manual_seed(seed)
random.seed(seed)
assert train_batch_size % gradient_accumulation_steps == 0, ("train_batch_size must be divisible by gradient_accumulation_steps")
micro_train_batch_size = train_batch_size // gradient_accumulation_steps    
assert rollout_batch_size % group_size == 0, ("rollout_batch_size must be divisible by group_size")
n_prompts_per_rollout_batch = rollout_batch_size // group_size    
assert train_batch_size >= group_size, ("train_batch_size must be greater than or equal to group_size")
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size    
print(f"Training configuration:")
print(f"micro_train_batch_size: {micro_train_batch_size}")
print(f"n_prompts_per_rollout_batch: {n_prompts_per_rollout_batch}")
print(f"n_microbatches_per_rollout_batch: {n_microbatches_per_rollout_batch}")
    
wandb.init(
    project=args.wandb_project,  
    name=f"grpo_{args.loss_type}_bs{args.rollout_batch_size}_lr{args.learning_rate}",
    config={
        "n_grpo_steps": args.n_grpo_steps,
        "advantage_eps": args.advantage_eps,
        "rollout_batch_size": args.rollout_batch_size,
        "group_size": args.group_size,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "loss_type": args.loss_type,
        "use_std_normalization": args.use_std_normalization,
        "use_length_normalization": args.use_length_normalization,
        "learning_rate": args.learning_rate,
        "sampling_temperature": args.sampling_temperature,
        "sampling_max_tokens": args.sampling_max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
)

wandb.define_metric("train_step") 
wandb.define_metric("eval_step") 
wandb.define_metric("train/*", step_metric="train_step")
wandb.define_metric("eval/*", step_metric="eval_step")


def sample_batch(dataset, batch_size):
    indices = torch.randperm(len(dataset))[:batch_size]
    return [dataset[i] for i in indices]


def data_tokenizer(dataset, model, tokenizer):
    """Tokenize dataset using the provided tokenizer"""
    data_tokenized = tokenize_prompt_and_output(
        [d["prompt"] for d in dataset],
        [d["response"] for d in dataset],
        tokenizer
    )
    
    input_ids = data_tokenized["input_ids"].to(model.device)
    labels = data_tokenized["labels"].to(model.device)
    response_mask = data_tokenized["response_mask"].to(model.device)
    return input_ids, labels, response_mask
    


def rollouts(policy, vllm_model, dataset, reward_fn, group_size, advantage_eps, loss_type, normalized_by_std, tokenizer):
    prompts = [data["prompt"] for data in dataset]
    gt_answers = [data["answer"] for data in dataset]
    load_policy_into_vllm_instance(policy, vllm_model)

    sampling_params = SamplingParams(
        temperature=sampling_temperature,  
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        n=group_size,
        include_stop_str_in_output=True
    )

    outputs = vllm_model.generate(prompts, sampling_params)
    repeated_ground_truths = [gt for gt in gt_answers for _ in range(group_size)]

    rollout_responses = []
    qa_pairs = []
    for output in outputs:
        prompt = output.prompt
        for response in output.outputs:
            rollout_responses.append(response.text)
            qa_pairs.append({"prompt": prompt, "response": response.text})

    advantages, raw_rewards, _ = compute_group_normalized_rewards(
        reward_fn, rollout_responses, repeated_ground_truths, 
        group_size, advantage_eps, normalized_by_std
    )
    input_ids, labels, response_mask = data_tokenizer(qa_pairs, policy, tokenizer)

    if loss_type == "grpo_clip":
        with torch.inference_mode():
            old_log_probs = get_response_log_probs(
                policy, 
                input_ids.to(policy.device), 
                labels.to(policy.device)
            )["log_probs"]
    else:
        old_log_probs = torch.empty(input_ids.shape, dtype=torch.float32).to(policy.device)

    return (
        advantages.to(policy.device), 
        raw_rewards.to(policy.device), 
        old_log_probs.to(policy.device),
        input_ids.to(policy.device), 
        labels.to(policy.device), 
        response_mask.to(policy.device), 
        qa_pairs
    )


def training(dataset_path, vllm_model, policy, reward_fn, lr, group_size, advantage_eps, 
             use_std_normalization, gradient_accumulation_steps, loss_type, clip_range,
             use_length_normalization, max_tokens_train, tokenizer):
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    dataset = load_dataset(dataset_path)
    dataset_formatted = dataset_f(dataset, args.prompt_path)
    
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    micro_train_batch_size = args.train_batch_size // gradient_accumulation_steps
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size
    
    total_samples_processed = 0
    
    for grpo_iteration in range(args.n_grpo_steps):
        print(f"\nGRPO iteration {grpo_iteration + 1}/{args.n_grpo_steps}")
        
        batch_data = sample_batch(dataset_formatted, n_prompts_per_rollout_batch)
        
        advantages, raw_rewards, old_log_probs, input_ids, labels, response_mask, qa_pairs = rollouts(
            policy, 
            vllm_model, 
            batch_data,  
            reward_fn, 
            group_size, 
            advantage_eps,
            loss_type,  
            use_std_normalization,
            tokenizer
        )
        
        print(f"Generated {len(qa_pairs)} rollouts")
        print(f"Sample rewards: {raw_rewards[:5].tolist()}")
        
        for epoch in range(args.epochs_per_rollout_batch):
            print(f"  Epoch {epoch + 1}/{args.epochs_per_rollout_batch}")
            
            indices = torch.randperm(args.rollout_batch_size)   
            rollout_input_ids = input_ids[indices]
            rollout_labels = labels[indices]
            rollout_response_mask = response_mask[indices]
            rollout_raw_rewards = raw_rewards[indices]
            rollout_old_log_probs = old_log_probs[indices]
            rollout_advantages = advantages[indices]

            epoch_loss = 0.0
            
            for step in range(n_microbatches_per_rollout_batch):
                total_samples_processed += micro_train_batch_size
                
                start_idx = step * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                
                batch_input_ids = rollout_input_ids[start_idx:end_idx]
                batch_labels = rollout_labels[start_idx:end_idx]
                batch_response_mask = rollout_response_mask[start_idx:end_idx]
                batch_advantages = rollout_advantages[start_idx:end_idx]
                batch_old_log_probs = rollout_old_log_probs[start_idx:end_idx]
                batch_raw_rewards = rollout_raw_rewards[start_idx:end_idx]

                policy_log_probs = get_response_log_probs(
                    policy, 
                    batch_input_ids, 
                    batch_labels
                )['log_probs']
                
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs, 
                    batch_response_mask, 
                    gradient_accumulation_steps, 
                    loss_type, 
                    batch_raw_rewards, 
                    batch_advantages, 
                    batch_old_log_probs, 
                    clip_range,
                    use_length_normalization,
                    max_tokens_train
                )
                epoch_loss += loss.item()

                if total_samples_processed % args.train_batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"    Optimizer step at {total_samples_processed} samples")
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/total_samples": total_samples_processed,
                    "train/rewards_mean": batch_raw_rewards.mean().item(),
                    "train/advantages_mean": batch_advantages.mean().item(),
                })
            
            avg_loss = epoch_loss / n_microbatches_per_rollout_batch
            print(f"    Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
            
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/grpo_iteration": grpo_iteration,
            })


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

policy = AutoModelForCausalLM.from_pretrained(model_path).to(device)
print(f"policy device: {policy.device}")

vllm_model = init_vllm(model_path, seed, gpu_memory_utilization)



training(
    dataset_path=train_ds_path,
    vllm_model=vllm_model,
    policy=policy,
    reward_fn=r1_zero_reward_fn,
    lr=lr,
    group_size=group_size,
    advantage_eps=advantage_eps,
    use_std_normalization=use_std_normalization,
    gradient_accumulation_steps=gradient_accumulation_steps,
    loss_type=loss_type,
    clip_range=clip_range,
    use_length_normalization=use_length_normalization,
    max_tokens_train=sampling_max_tokens,
    tokenizer=tokenizer
)


wandb.finish()