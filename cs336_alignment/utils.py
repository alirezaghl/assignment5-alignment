import torch
import vllm
import json
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import SamplingParams, LLM
from transformers import PreTrainedModel
from unittest.mock import patch
import re



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