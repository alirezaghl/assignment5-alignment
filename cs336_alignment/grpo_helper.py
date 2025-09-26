from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

def compute_group_normailzed_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
):

    raw_rewards = [reward_fn(prompt, gt)['reward'] for prompt, gt in zip(rollout_responses, repeated_ground_truths)]
    group_rewards = [raw_rewards[i:i + group_size] for i in range(0, len(raw_rewards), group_size)]

    advantages = []

    for i, group_reward in enumerate(group_rewards):
        group_reward = torch.tensor(group_reward)
        mean = torch.mean(group_reward)
        std = torch.std(group_reward)

        for reward in group_reward:
            advantage = reward - mean
            if normalize_by_std:
                advantage /= (std) + advantage_eps
            advantages.append(advantage)

    advantages = torch.Tensor(advantages)
    raw_rewards = torch.Tensor(raw_rewards)
    metadata = {}
    metadata['advantages'] = advantages
    metadata['raw_rewards'] = raw_rewards

    return (advantages, raw_rewards, metadata)

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages,
        policy_log_probs
):
    pg_loss = -raw_rewards_or_advantages.expand_as(policy_log_probs) * policy_log_probs
    return pg_loss


def compute_grpo_clip_loss(
        advantages,
        policy_log_probs,
        old_log_probs,
        cliprange
):
    advantages = advantages.expand_as(policy_log_probs)

    improvement = torch.exp(policy_log_probs - old_log_probs)

    arg_1 = improvement * advantages

    arg_2 = torch.clip(improvement, 1.0-cliprange, 1.0+cliprange) * advantages

    loss = -torch.min(arg_1, arg_2)


    clip_mask = (arg_1 != arg_2)

    clipped_percentage = clip_mask.sum() / arg_2.numel()

    metadata = {}

    metadata["clipped_percentage"] = clipped_percentage


    return (loss, metadata)

def compute_policy_gradient_loss(
    policy_log_probs,
    loss_type,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange
):
    if loss_type == 'no_baseline':
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    
    elif loss_type == 'reinforce_with_baseline':
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    
    elif loss_type == 'grpo_clip':
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
    return (loss, metadata)


def masked_mean(tensor, mask, dim):
    if dim is None:
        mean = torch.mean(tensor[mask == 1])
    else:
        masked = torch.where(mask == 1, tensor, 0.0)
        mean = torch.sum(masked, dim=dim) /torch.sum(mask == 1, dim=dim)
    return mean

def grpo_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        clip_range
):
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs,
                                           clip_range)
    
    loss = masked_mean(loss, response_mask, dim=None)

    loss /= gradient_accumulation_steps

    loss.backward()


    return loss, metadata

