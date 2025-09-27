import torch
from vllm import SamplingParams, LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import json
import numpy as np
import sys
import importlib.util
from tqdm import tqdm
import argparse
import re
import random
from transformers import PreTrainedModel
from sft_helper import tokenize_prompt_and_output, get_response_log_probs, masked_normalize,sft_microbatch_train_step, compute_entropy
from drgrpo_grader import r1_zero_reward_fn
from utils import load_dataset, init_vllm, load_policy_into_vllm_instance, dataset_f



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, choices=[128, 256, 512, 1024, -1], 
                       default=-1, help="Number of training examples (-1 for full dataset)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs_sft", type=int, choices=[1,2,3,4],default=2)
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--n_ei_batch_sizes", type=int, choices=[512, 1024, 2048], default=512)
    parser.add_argument("--n_ei_rollouts", type=int, choices=[2,3,4,5,8], default=4, help="number of expert iteration rollouts")
    parser.add_argument("--train_path", type=str, default="/data/a5-alignment/gsm8k/train.jsonl")
    parser.add_argument('--test_path', type=str, default='/data/a5-alignment/gsm8k/test.jsonl')
    parser.add_argument("--filtered_data_path", type=str, default="./filtered_math.json")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    args = parser.parse_args()
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    train_path = args.train_path
    eval_path = args.test_path
    prompt_path = "./prompts/r1_zero.prompt"
    
    dataset_name = "full" if args.dataset_size == -1 else str(args.dataset_size)
    run_name = f"ei_math_db{args.n_ei_batch_sizes}_g{args.n_ei_rollouts}_epochs{args.n_epochs_sft}"    
    wandb.init(project="gsm8k-expert-iteration", name=run_name)
    
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    vllm_model = init_vllm(model_id, SEED, gpu_memory_utilization=0.2)
    
    train_dataset = load_dataset(train_path)
    eval_dataset = load_dataset(eval_path)
    
    train_dataset = dataset_f(train_dataset, prompt_path)
    test_dataset = dataset_f(eval_dataset, prompt_path)
    
    if args.dataset_size != -1:
        train_dataset = train_dataset[:args.dataset_size]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    def data_tokenizer(dataset):
         data_tokenized = tokenize_prompt_and_output(
              [d["prompt"] for d in dataset],
              [d["response"] for d in dataset],
              tokenizer
         )
         input_ids = data_tokenized["input_ids"].to(model.device)
         labels = data_tokenized["labels"].to(model.device)
         response_mask = data_tokenized["response_mask"].to(model.device)
         return input_ids, labels, response_mask
    
    def sample_batch(dataset, batch_size):
        indices = torch.randperm(len(dataset))[:batch_size]
        return [dataset[i] for i in indices]

    def evaluate_model(vllm_model, dataset, reward_fn):
        prompts = [data["prompt"] for data in dataset]
        gt_answers = [data["answer"] for data in dataset]

        sampling_params = SamplingParams(
            temperature=1.0,  
            top_p=1.0,
            max_tokens=1024,
            min_tokens=4,
            stop=["</answer>"],
            n=args.n_ei_rollouts,
            include_stop_str_in_output=True
        )
        
        outputs = vllm_model.generate(prompts, sampling_params)
        total_reward = 0
        format_reward = 0
        answer_reward = 0
        filtered_qa = []
        total_generated = 0
        
        for output, gt in zip(outputs, gt_answers):
            prompt = output.prompt
            for response in output.outputs:
                total_generated += 1
                response_text = response.text
                reward_dict = reward_fn(response_text, gt)
                if reward_dict["reward"] == 1:
                    filtered_qa.append({"prompt":prompt, "response":response_text, "answer":gt})
                    total_reward += reward_dict["reward"]
                    format_reward += reward_dict["format_reward"]
                    answer_reward += reward_dict["answer_reward"]

        n = len(filtered_qa)
        success_rate = n / max(total_generated, 1)
        print(f"generated {total_generated} examples, {n} correct (success rate: {success_rate:.3f})")
        
        if n == 0:
            return filtered_qa, {"reward":0, "format_reward":0, "answer_reward":0}
        return filtered_qa, {"reward":total_reward/n, "format_reward":format_reward/n, "answer_reward":answer_reward/n}
    
    def evaluate_on_test(vllm_model, test_dataset, reward_fn):
        sample_test = test_dataset[:100]  
        prompts = [data["prompt"] for data in sample_test]
        gt_answers = [data["answer"] for data in sample_test]

        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=1024,
            min_tokens=4,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
        
        outputs = vllm_model.generate(prompts, sampling_params)
        total_reward = 0
        format_reward = 0
        answer_reward = 0
        
        for output, gt in zip(outputs, gt_answers):
            response_text = output.outputs[0].text
            reward_dict = reward_fn(response_text, gt)
            total_reward += reward_dict["reward"]
            format_reward += reward_dict["format_reward"]
            answer_reward += reward_dict["answer_reward"]

        n = len(outputs)
        return {
            "total_accuracy": total_reward / n,
            "format_accuracy": format_reward / n,
            "answer_accuracy": answer_reward / n
        }

    def sft_trainer(dataset, input_ids, labels, response_mask):
        shuffle_indices = torch.randperm(len(dataset))
        total_loss = 0
        num_batches = len(dataset) // args.batch_size
        
        if num_batches == 0:
            return 0.0
        
        for epoch in range(args.n_epochs_sft):
            epoch_loss = 0
            for i in range(num_batches):
                start_idx = i * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(dataset))
                batch_indices = shuffle_indices[start_idx:end_idx]

                batch_input_ids = input_ids[batch_indices]
                batch_labels = labels[batch_indices]
                batch_response_mask = response_mask[batch_indices]
                
                with torch.no_grad():
                    outputs = model(input_ids=batch_input_ids, labels=batch_labels)
                    logits = outputs.logits
                    entropy = compute_entropy(logits).mean().item()
                        
                policy_log_probs = get_response_log_probs(model, batch_input_ids, batch_labels)["log_probs"]
                loss, _ = sft_microbatch_train_step(policy_log_probs, batch_response_mask, args.gradient_accumulation_steps)
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                wandb.log({
                    "train_loss": loss.item(),
                    "entropy": entropy,
                })
                        
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}/{args.n_epochs_sft}, Loss: {epoch_loss/num_batches:.4f}")
        
        return total_loss / (num_batches * args.n_epochs_sft)
    
    def expert_trainer(dataset):

        filtered_qa = []
        eval_results = {"reward": 0, "format_reward": 0, "answer_reward": 0}
        
        if len(dataset) == 0:
            return filtered_qa
        
        load_policy_into_vllm_instance(model, vllm_model)
        filtered_qa, eval_results = evaluate_model(vllm_model, dataset, r1_zero_reward_fn)
        
        print(f"expert iteration results:")
        print(f"  Reward: {eval_results['reward']:.3f}")
        print(f"  Format reward: {eval_results['format_reward']:.3f}")
        print(f"  Answer reward: {eval_results['answer_reward']:.3f}")
        
        return filtered_qa
    
    def bootstrap():

        print(f"expert iteration with {args.n_ei_steps} steps")
        print(f"batch size: {args.n_ei_batch_sizes}, n_rollouts: {args.n_ei_rollouts}")
        
        all_filtered_examples = []
        
        for ei_step in range(args.n_ei_steps):
            print(f"\nexpert iteration epoch {ei_step + 1}")
            
            dataset_batch = sample_batch(train_dataset, args.n_ei_batch_sizes)
            
            new_filtered_examples = expert_trainer(dataset_batch)
            
            if len(new_filtered_examples) == 0:
                print("no correct response")
                continue
            
            all_filtered_examples.extend(new_filtered_examples)
            print(f"ei Step {ei_step + 1}")
            print(f"n_accumulated examples: {len(all_filtered_examples)}")
            
            print("SFT phase")
            input_ids, labels, response_mask = data_tokenizer(all_filtered_examples)
            avg_loss = sft_trainer(all_filtered_examples, input_ids, labels, response_mask)
            
            print("evaluation")
            load_policy_into_vllm_instance(model, vllm_model)
            test_results = evaluate_on_test(vllm_model, test_dataset, r1_zero_reward_fn)
            
            print(f"  accuracy: {test_results['total_accuracy']:.3f}")
            print(f"  format accuracy: {test_results['format_accuracy']:.3f}")
            print(f"  answer accuracy: {test_results['answer_accuracy']:.3f}")
            
            wandb.log({
                "ei_step": ei_step,
                "avg_train_loss": avg_loss,
                "test_total_accuracy": test_results['total_accuracy'],
                "test_format_accuracy": test_results['format_accuracy'],
                "test_answer_accuracy": test_results['answer_accuracy'],
                "new_filtered_examples": len(new_filtered_examples),
                "total_filtered_examples": len(all_filtered_examples),
                "batch_size": args.n_ei_batch_sizes,
                "rollouts": args.n_ei_rollouts,
            })
        
        print(f"\nSaving {len(all_filtered_examples)} filtered examples to {args.filtered_data_path}")
        with open(args.filtered_data_path, "w") as f:
            json.dump({
                "dataset": all_filtered_examples,
                "metadata": {
                    "total_examples": len(all_filtered_examples),
                    "ei_steps": args.n_ei_steps,
                    "batch_size": args.n_ei_batch_sizes,
                    "rollouts": args.n_ei_rollouts,
                    "sft_epochs": args.n_epochs_sft,
                }
            }, f, indent=4)
        
    
    bootstrap()

if __name__ == "__main__":
    main()