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
from transformers import PreTrainedModel
from sft_helper import tokenize_prompt_and_output, get_response_log_probs, masked_normalize,sft_microbatch_train_step
from drgrpo_grader import r1_zero_reward_fn



def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def init_vllm(model_id, seed, gpu_memory_utilization=0.1):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            max_model_len=1024,
            enforce_eager=True,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate_model(vllm_model, dataset, reward_fn, max_tokens=1024):
    prompts = [data["prompt"] for data in dataset]
    gt_answers = [data["answer"] for data in dataset]
    
    sampling_params = SamplingParams(
        temperature=1.0,  
        top_p=1.0,
        max_tokens=max_tokens,
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
    
    n = len(dataset)
    return {
        "total_accuracy": total_reward / n,
        "format_accuracy": format_reward / n,
        "answer_accuracy": answer_reward / n
    }

def parse_gsm8k(answer_text):
    match = re.search(r"####\s*(.+?)(?:\s*$)", answer_text, re.MULTILINE)
    
    if match:
        final_answer = match.group(1).strip()
        reasoning = answer_text[:match.start()].strip()
    else:
        lines = answer_text.splitlines()
        final_answer = lines[-1].strip()
        reasoning = "\n".join(lines[:-1]).strip()
    
    return final_answer, reasoning

def dataset_f(dataset, prompt_path):
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    formatted_data = []
    for example in dataset:
        question = example["question"].strip()
        answer_text = example["answer"].strip()
        
        final_answer, reasoning = parse_gsm8k(answer_text)
        
        prompt = prompt_template.format(question=question)
        response = f"<think>\n{reasoning}\n</think> <answer>{final_answer}</answer>"
        
        formatted_data.append({
            "prompt": prompt,
            "response": response, 
            "answer": final_answer
        })
    
    return formatted_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, choices=[128, 256, 512, 1024, -1], 
                       default=-1, help="Number of training examples (-1 for full dataset)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--train_path", type=str, default="/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/train.jsonl")
    parser.add_argument('--test_path', type=str, default='/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/test.jsonl')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    
    args = parser.parse_args()
    
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    train_path = args.train_path
    eval_path = args.test_path
    prompt_path = "/content/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    
    dataset_name = "full" if args.dataset_size == -1 else str(args.dataset_size)
    run_name = f"sft_{dataset_name}_lr{args.lr}_bs{args.batch_size}"    
    wandb.init(project="gsm8k-sft-experiment", name=run_name)
    
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # working with single A100 gpu of colab :(
    vllm_model = init_vllm(model_id, SEED, gpu_memory_utilization=0.1)
    
    train_dataset = load_dataset(train_path)
    eval_dataset = load_dataset(eval_path)
    
    if args.dataset_size != -1:
        train_dataset = train_dataset[:args.dataset_size]
    
    train_dataset = dataset_f(train_dataset, prompt_path)
    test_dataset = dataset_f(eval_dataset, prompt_path)
    
    train_tokenized = tokenize_prompt_and_output(
        [d["prompt"] for d in train_dataset],
        [d["response"] for d in train_dataset],
        tokenizer
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    train_input_ids = train_tokenized["input_ids"].to(model.device)
    train_labels = train_tokenized["labels"].to(model.device)
    train_response_mask = train_tokenized["response_mask"].to(model.device)
    
    
    for epoch in range(args.num_epochs):
        print(f"\nepoch {epoch + 1}")
        
        shuffle_indices = torch.randperm(len(train_dataset))
        total_loss = 0
        num_batches = len(train_dataset) // args.batch_size
        
        for i in tqdm(range(num_batches), desc=f"epoch {epoch + 1}"):
            start_idx = i * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(train_dataset))
            batch_indices = shuffle_indices[start_idx:end_idx]
            
            input_ids = train_input_ids[batch_indices]
            labels = train_labels[batch_indices]
            response_mask = train_response_mask[batch_indices]
            
            policy_log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, args.gradient_accumulation_steps)
            total_loss += loss.item()
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        print(f"training loss: {avg_loss:.3f}")
        
        print("evaluation")
        with torch.no_grad():
            load_policy_into_vllm_instance(model, vllm_model)
            # preferred not use log generations!
            eval_results = evaluate_model(vllm_model, test_dataset, r1_zero_reward_fn, max_tokens=1024)
            
            print(f"Validation Results:")
            print(f"  Total Accuracy: {eval_results['total_accuracy']:.3f}")
            print(f"  Format Accuracy: {eval_results['format_accuracy']:.3f}")
            print(f"  Answer Accuracy: {eval_results['answer_accuracy']:.3f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_total_accuracy": eval_results['total_accuracy'],
                "val_format_accuracy": eval_results['format_accuracy'],
                "val_answer_accuracy": eval_results['answer_accuracy'],
                "dataset_size": len(train_dataset)
            })
    
    print(f"validation accuracy: {eval_results['total_accuracy']:.3f}")

if __name__ == "__main__":
        main()