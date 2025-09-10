import json

def rewards(file_path):
    both_correct = 0
    both_wrong = 0
    format_wrong_answer_correct = 0
    format_correct_answer_wrong = 0
    total_examples = 0
    
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    for result in results:
        reward = result['reward']
        format_reward = reward['format_reward']
        answer_reward = reward['answer_reward']
        
        total_examples += 1
        
        if format_reward == 1.0 and answer_reward == 1.0:
            both_correct += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            both_wrong += 1
        elif format_reward == 0.0 and answer_reward == 1.0:
            format_wrong_answer_correct += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            format_correct_answer_wrong += 1
    
    print(f"total: {total_examples}")
    print(f"both correct: {both_correct}")
    print(f"both wrong: {both_wrong}")
    print(f"format wrong, answer correct: {format_wrong_answer_correct}")
    print(f"format correct, answer wrong: {format_correct_answer_wrong}")

file_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/assignment5-alignment/cs336_alignment/baseline_results.json"
rewards(file_path)