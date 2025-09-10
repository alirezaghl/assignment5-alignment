import torch
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F
import json
from gsm8k_baseline import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn

r1_zero_prompt_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
validation_examples_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/data/gsm8k/test.jsonl"



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    all_sequences = []
    mask_sequences = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
        output_tokens = tokenizer(output, add_special_tokens=False)['input_ids']
        
        full_sequence = prompt_tokens + output_tokens
        all_sequences.append(full_sequence)
        
        mask = [0] * len(prompt_tokens) + [1] * len(output_tokens)
        mask_sequences.append(mask)
    
    max_len = max(len(seq) for seq in all_sequences)
    
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(all_sequences, mask_sequences):
        padded_seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
        
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_masks.append(padded_mask)
    
    input_ids = torch.tensor(padded_sequences, dtype=torch.long)
    response_mask = torch.tensor(padded_masks, dtype=torch.bool)
        
    return {
        'input_ids': input_ids[:, :-1],
        'labels': input_ids[:, 1:], 
        'response_mask': response_mask[:, 1:]
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the logits (i.e., entropy of the final dimension).
    
    Args:
        logits: shape (..., vocab_size) - can be any shape with vocab_size as last dim
    
    Returns:
        entropy: shape (...) - same shape as input minus last dimension
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)

    ret_dict = {}
    ret_dict['log_probs'] = log_probs

    if return_token_entropy:
        ret_dict['token_entropy'] = compute_entropy(logits)
    
    return ret_dict


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask.float()
    
    if dim is None:
        masked_sum = torch.sum(masked_tensor)
    else:
        masked_sum = torch.sum(masked_tensor, dim=dim)
    
    return masked_sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the 
                         SFT policy being trained.
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: The constant by which to divide the sum. It is fine to leave this as 1.0.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation.
            - metadata: Dict with metadata from the underlying loss call, and any other statistics.
    """

    response_nll = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()

    microbatch_loss = response_nll / gradient_accumulation_steps

    microbatch_loss.backward()

    metadata = {}
    metadata['loss'] = microbatch_loss.item()  

    return (microbatch_loss, metadata)



def log_generations(eval_model, policy_model, reward_fn, tokenizer, batch_size):
    sampling_params = SamplingParams(
        temperature=1.0,  
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    with open(r1_zero_prompt_path, 'r') as f: 
        r1_prompt = f.read()
    
    with open(validation_examples_path, 'r') as f:
        prompts = [json.loads(line) for line in f]  
    
    prompts_f = []  
    answers = []
    
    for prompt in prompts:
        prompt_f = r1_prompt.format(question=prompt.get('question'))
        prompts_f.append(prompt_f)
        answers.append(prompt.get('answer'))
    
    results = evaluate_vllm(eval_model, reward_fn, prompts_f, answers, sampling_params)
    
    responses = [result['response'] for result in results]
    ground_truths = [result['gt'] for result in results]
    rewards = [result['reward']['reward'] for result in results]
    
    tokenization_dict = tokenize_prompt_and_output(prompts_f, responses, tokenizer)
    
    input_ids = tokenization_dict['input_ids'] 
    response_mask = tokenization_dict['response_mask']
    
    response_length = response_mask.sum(dim=-1)
    
    avg_token_entropies = []
    
    for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i+batch_size, :]
            batch_response_mask = response_mask[i:i+batch_size, :]
            logits = policy_model(batch_input_ids).logits
            token_entropy = compute_entropy(logits)
            masked_entropy = token_entropy * batch_response_mask.float()
            response_token_counts = torch.sum(batch_response_mask.float(), dim=-1)
            batch_avg_entropy = torch.sum(masked_entropy, dim=-1) / (response_token_counts + 1e-8)
            avg_token_entropies.extend(batch_avg_entropy.tolist())
  
    
    flag = torch.tensor([int(reward == 1.0) for reward in rewards])
    correct_indices = torch.where(flag == 1)[0]
    incorrect_indices = torch.where(flag == 0)[0]
    
    avg_len = response_length.float.mean().item()
    avg_correct_length = response_length[correct_indices].float.mean().item() if len(correct_indices) > 0 else float("nan")
    avg_incorrect_length = response_length[incorrect_indices].float.mean().item() if len(incorrect_indices) > 0 else float("nan")
    
    
    return {
        'responses': responses,
        'ground_truths': ground_truths,
        'rewards': rewards,
        'response_lengths': response_length.tolist(),
        'avg_response_length': avg_len,
        'avg_correct_length': avg_correct_length,
        'avg_incorrect_length': avg_incorrect_length,
        'avg_token_entropy': avg_token_entropies
    }