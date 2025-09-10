from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import json
from collections.abc import Callable
from typing import List
from drgrpo_grader import r1_zero_reward_fn

def format(response):
    response = re.sub(r'</think>\s*<answer>', '</think> <answer>', response)
    return response


def evaluate_vllm(
 vllm_model: LLM,
 reward_fn: Callable[[str,str], dict[str, float]],
 prompts:List[str],
 answers,
 eval_sampling_params:SamplingParams
 )-> None:
 """
 Evaluate a language model on a list of prompts,
 compute evaluation metrics.
 """
 outputs = vllm_model.generate(prompts, eval_sampling_params)

 results = []
 for output, gt in zip(outputs, answers):
  prompt = output.prompt
  response = format(output.outputs[0].text)
  reward = reward_fn(response, gt)
  results.append({"prompt": prompt, "gt": gt, "response": response, "reward": reward})
 
 return results


def evaluate_Qwen2():
    model_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/models/Qwen2.5-Math-1.5B"
    validation_examples_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/test.jsonl"
    r1_zero_prompt_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    sampling_params.stop =["</answer>"]
    sampling_params.include_stop_str_in_output=True

    llm = LLM(model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            dtype="bfloat16")

    with open(r1_zero_prompt_path, 'r') as f:
        r1_prompt = f.read()

    with open(validation_examples_path, 'r') as f:
        prompts = [json.loads(_) for _ in f]

    prompts_f = []
    answers = []

    for prompt in prompts:
        prompt_f = r1_prompt.format(
                question=prompt.get('question')
            )
        prompts_f.append(prompt_f)
        answers.append(prompt.get('answer'))
    
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts_f, answers, sampling_params)
    file_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/baseline_results.json"
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"results saved!")

evaluate_Qwen2()