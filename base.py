import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import time
import re
from tqdm import tqdm
from collections import Counter
import psutil
import numpy as np
from typing import Dict, List, Optional
from contextlib import contextmanager

class Config:
    def __init__(self):
        self.batch_idx = 0
        self.baseline = True
        self.device_batch_size = 6
        self.max_idx = 128
        self.n_votes = 4
        self.temp = 1.0
        self.start_final_answer_idx = 384
        self.answer_length = 12
        self.root_prefix = "/scratch/mwa7459/"
        self.checkpoint = "mistralai/Mistral-7B-v0.1"
        self.final_answer_text = "\nTherefore, the answer (arabic numerals) is"
        self.zero_shot_cot_prompt = "\nA: Let's think step by step."
        self.n_ahead = 8

class TestMetrics:
    def __init__(self):
        self.total_questions = 0
        self.correct_answers = 0
        self.predictions = []
        self.ground_truths = []
        
    def update(self, prediction, ground_truth):
        if prediction is not None and ground_truth is not None:
            self.total_questions += 1
            is_correct = prediction == ground_truth
            self.correct_answers += int(is_correct)
            self.predictions.append(prediction)
            self.ground_truths.append(ground_truth)
        
    def get_accuracy(self):
        return self.correct_answers / self.total_questions if self.total_questions > 0 else 0

def get_memory_usage():
    """Get current memory usage in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return psutil.Process().memory_info().rss

def update_model_config(config, model):
    """Update config with actual model parameters."""
    config.d_model = model.config.hidden_size
    config.n_layers = model.config.num_hidden_layers
    config.n_heads = model.config.num_attention_heads
    config.d_ff = model.config.intermediate_size
    config.n_ctx = model.config.max_position_embeddings

def calculate_flops_per_token(config) -> int:
    """Calculate FLOPs per token using actual model parameters."""
    N = 2 * config.d_model * config.n_layers * (
        2 * (config.d_model // config.n_heads) + config.d_ff
    )
    C_forward = 2 * N + 2 * config.n_layers * config.d_model
    return C_forward

def extract_first_integer(s):
    """Extract the first integer from a string."""
    if not s:
        return None
    match = re.search(r'\d+', str(s).replace(',', ''))
    if match:
        return int(match.group())
    return None

def model_init(args, params=None):
    """Initialize the model with given parameters."""
    if params is None:
        params = {}
    
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        cache_dir=args.root_prefix + "cache"
    )
    print("Loaded model")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.tokenizer = tokenizer
    model.config_params = params
    model.eval_mode = True
    model.eval()
    return model

def calculate_cumulative_accuracy(answers, true_answer):
    """Calculate accuracy based on majority voting of answers so far."""
    if not answers:
        return 0.0
    answer_counts = Counter([a for a in answers if a is not None])
    if not answer_counts:
        return 0.0
    majority_answer = answer_counts.most_common(1)[0][0]
    return float(majority_answer == true_answer)

def save_vote_metrics(filepath: str, outputs: str, flops: int, time_taken: float, 
                     extracted: Counter, true_answer: int, correct: bool, 
                     seq_length: int, flops_per_token: int, 
                     cumulative_answers: List[Optional[int]], 
                     vote_idx: int):
    """Save metrics for a single vote."""
    # Calculate accuracy up to this vote
    cumulative_accuracy = calculate_cumulative_accuracy(cumulative_answers, true_answer)
    
    with open(filepath, 'w') as f:
        f.write(f"{outputs}\n\n")
        f.write(f"=== Vote Metrics ===\n")
        f.write(f"Total FLOPs: {flops:,}\n")
        f.write(f"Time Taken: {time_taken:.2f} seconds\n")
        f.write(f"Extracted: {dict(extracted)}\n")
        f.write(f"True Answer: {true_answer}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"FLOPs/token: {flops_per_token:,}\n")
        f.write(f"\n=== Accuracy Metrics ===\n")
        f.write(f"Current Vote: {vote_idx}\n")
        f.write(f"Answers so far: {cumulative_answers}\n")
        f.write(f"Cumulative accuracy: {cumulative_accuracy:.2%}\n")

def save_question_metrics(filepath: str, question_idx: int, all_votes: List[Dict], 
                        total_flops: int, total_time: float, final_accuracy: float,
                        true_answer: int):
    """Save aggregated metrics for a question."""
    with open(filepath, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Question {question_idx} Summary:\n")
        f.write(f"True Answer: {true_answer}\n\n")
        
        # Calculate cumulative metrics for each vote
        cumulative_answers = []
        f.write("Vote Progression:\n")
        for vote_idx, vote in enumerate(all_votes, 1):
            cumulative_answers.append(vote['answer'])
            current_accuracy = calculate_cumulative_accuracy(cumulative_answers, true_answer)
            
            f.write(f"Vote {vote_idx}:\n")
            f.write(f"  Answer: {vote['answer']}\n")
            f.write(f"  FLOPs: {vote['flops']:,}\n")
            f.write(f"  Time: {vote['time']:.2f}s\n")
            f.write(f"  Accuracy up to vote {vote_idx}: {current_accuracy:.2%}\n")
            
            # If this vote changed the majority answer, note it
            if vote_idx > 1:
                prev_accuracy = calculate_cumulative_accuracy(cumulative_answers[:-1], true_answer)
                if prev_accuracy != current_accuracy:
                    f.write(f"  >>> Majority answer changed at vote {vote_idx}\n")
            f.write("\n")
        
        f.write("\nQuestion Totals:\n")
        f.write(f"Total FLOPs: {total_flops:,}\n")
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write(f"Final Question Accuracy: {final_accuracy:.2%}\n")
        f.write(f"Vote Distribution: {dict(Counter(cumulative_answers))}\n")

def main():
    # Set random seeds for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Initialize configuration and model
    args = Config()
    model = model_init(args)
    update_model_config(args, model)
    flops_per_token = calculate_flops_per_token(args)
    print(f"FLOPs per token: {flops_per_token:,}")
    
    # Initialize dataset and metrics
    cot_dataset_gsm = load_dataset("gsm8k", "main", split="test").shuffle(seed=random_seed)
    metrics = TestMetrics()
    metrics_file = f"answers/metrics_{'baseline' if args.baseline else 'ft'}_{args.n_ahead if not args.baseline else 1}_{args.temp}_{args.n_votes}.txt"
    
    # Ensure output directories exist
    os.makedirs("answers", exist_ok=True)
    
    # Calculate batch ranges
    start_question = args.device_batch_size * args.batch_idx
    end_question = min(args.max_idx, args.device_batch_size * (args.batch_idx + 1))
    print(f"Processing questions from {start_question} to { args.max_idx}")
    
    # Main inference loop
    for question_idx in tqdm(range(start_question,  args.max_idx)):
        question_start_time = time.perf_counter()
        question_total_flops = 0
        all_vote_results = []
        
        # Skip if already processed
        last_save_folder = f"answers/eval_{'baseline' if args.baseline else 'ft'}_{args.n_ahead if not args.baseline else 1}_{args.temp}_{args.n_votes}"
        if os.path.exists(last_save_folder + f"/{question_idx}.txt"):
            print(f"Skipping question {question_idx}")
            continue
        
        # Get the question
        try:
            question_data = cot_dataset_gsm[question_idx]
            true_answer = extract_first_integer(question_data["answer"].split("#### ")[-1])
            if true_answer is None:
                print(f"Warning: Could not extract true answer for question {question_idx}")
                continue
        except Exception as e:
            print(f"Error accessing question {question_idx}: {str(e)}")
            continue
        
        extracted_answers = []
        for vote_idx in range(1, args.n_votes + 1):
            vote_start_time = time.perf_counter()
            vote_flops = 0
            
            # Create output directory
            folder_name = f"answers/eval_{'baseline' if args.baseline else 'ft'}_{args.n_ahead if not args.baseline else 1}_{args.temp}_{vote_idx}"
            os.makedirs(folder_name, exist_ok=True)
            
            # Prepare input
            input_text = f"Q: {question_data['question']}{args.zero_shot_cot_prompt}"
            inputs = model.tokenizer([input_text], return_tensors="pt", padding=True).to(model.device)
            
            # Generate solution
            with torch.no_grad():
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
                started_generating_answer_at = None
                
                for cur_token_idx in range(args.start_final_answer_idx + args.answer_length):
                    # Generate next token
                    new_ids = model(
                        input_ids[~finished_generating],
                        attention_mask=attention_mask[~finished_generating]
                    )['logits']
                    
                    # Update FLOPs count for this step
                    vote_flops += flops_per_token * (~finished_generating).sum()
                    
                    # Process each sequence
                    for list_idx, answer_idx in enumerate((~finished_generating).nonzero(as_tuple=True)[0]):
                        base_answer_ids = input_ids[answer_idx]
                        new_answer_ids = new_ids[list_idx]
                        last_token_idx = (base_answer_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0].max()
                        
                        # Sample next token
                        if args.temp == 0:
                            new_ids_sampled = torch.argmax(new_answer_ids[last_token_idx]).unsqueeze(0)
                        else:
                            new_ids_sampled = torch.multinomial(
                                torch.nn.functional.softmax(new_answer_ids[last_token_idx] / args.temp, dim=-1),
                                1
                            )
                        
                        # Extend sequences if needed
                        if last_token_idx + 1 >= len(base_answer_ids):
                            new_padding = torch.full((len(input_ids), 1), model.tokenizer.pad_token_id,
                                                   dtype=torch.long, device=input_ids.device)
                            input_ids = torch.cat([input_ids, new_padding], dim=-1)
                            attention_mask = torch.cat([attention_mask, torch.zeros_like(new_padding)], dim=-1)
                        
                        attention_mask[answer_idx, last_token_idx + 1] = 1
                        input_ids[answer_idx, last_token_idx + 1] = new_ids_sampled
                        
                        # Check for completion
                        if new_ids_sampled in [model.tokenizer.eos_token_id,
                                             model.tokenizer.bos_token_id,
                                             model.tokenizer.pad_token_id]:
                            finished_generating[answer_idx] = 1
                        
                        # Clean up repeated text
                        decoded = model.tokenizer.decode(input_ids[answer_idx], skip_special_tokens=True)
                        for end_str in ["Q:", "\n\n\n"]:
                            if decoded.count(end_str) > 1:
                                decoded = decoded.split(end_str)[:-1]
                                new_answer = model.tokenizer.encode(decoded, return_tensors="pt").to(input_ids.device)
                                input_ids[answer_idx] = torch.ones_like(input_ids[answer_idx]) * model.tokenizer.pad_token_id
                                input_ids[answer_idx, :new_answer.shape[1]] = new_answer
                                attention_mask[answer_idx] = (input_ids[answer_idx] != model.tokenizer.pad_token_id).long()
                                finished_generating[answer_idx] = 1
                                break
                    
                    # Handle final answer generation
                    if ((cur_token_idx == args.start_final_answer_idx and started_generating_answer_at is None)
                        or finished_generating.all()):
                        if started_generating_answer_at is None:
                            finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
                            started_generating_answer_at = cur_token_idx
                            base_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                            final_texts = [text.rstrip() + args.final_answer_text for text in base_texts]
                            encoded_final_texts = model.tokenizer(final_texts, return_tensors="pt", padding=True).to(input_ids.device)
                            attention_mask = encoded_final_texts.attention_mask
                            input_ids = encoded_final_texts.input_ids
                        else:
                            break
                    
                    if started_generating_answer_at is not None:
                        if cur_token_idx - started_generating_answer_at > args.answer_length:
                            break
            
            # Process results
            decoded_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            vote_extracted_number = extract_first_integer(decoded_text.split(args.final_answer_text)[-1])
            extracted_answers.append((vote_extracted_number, true_answer, decoded_text))
            
            # Calculate vote metrics
            vote_time = time.perf_counter() - vote_start_time
            extracted_number = Counter([vote_extracted_number] if vote_extracted_number is not None else [])
            extracted_most_common = extracted_number.most_common(1)[0][0] if extracted_number else None
            correct = extracted_most_common == true_answer if extracted_most_common is not None else False
            
            # Calculate cumulative answers for this vote
            cumulative_answers = [ans[0] for ans in extracted_answers]
            
            # Save vote metrics
            save_vote_metrics(
                filepath=f"{folder_name}/{question_idx}.txt",
                outputs=decoded_text,
                flops=vote_flops,
                time_taken=vote_time,
                extracted=extracted_number,
                true_answer=true_answer,
                correct=correct,
                seq_length=input_ids.shape[1],
                flops_per_token=flops_per_token,
                cumulative_answers=cumulative_answers,
                vote_idx=vote_idx
            )
            
            # Update tracking
            question_total_flops += vote_flops
            all_vote_results.append({
                'answer': vote_extracted_number,
                'flops': vote_flops,
                'time': vote_time
            })
            
            # Clean up CUDA memory after each vote
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate and save final question metrics
        question_time = time.perf_counter() - question_start_time
        
        # Get final prediction using majority voting
        final_extracted = Counter([ans[0] for ans in extracted_answers if ans[0] is not None])
        final_answer = final_extracted.most_common(1)[0][0] if final_extracted else None
        
        # Update overall metrics
        metrics.update(final_answer, true_answer)
        
        # Save question metrics
        save_question_metrics(
            filepath=metrics_file,
            question_idx=question_idx,
            all_votes=all_vote_results,
            total_flops=question_total_flops,
            total_time=question_time,
            final_accuracy=metrics.get_accuracy(),
            true_answer=true_answer
        )
        
        # Print progress update
        print(f"\nQuestion {question_idx} completed:")
        print(f"Total FLOPs: {question_total_flops:,}")
        print(f"Total time: {question_time:.2f} seconds")
        print(f"Prediction: {final_answer}, True Answer: {true_answer}")
        print(f"Current accuracy: {metrics.get_accuracy():.2%}")
        
        # Clean up memory after each question
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print final summary
    print("\nInference completed!")
    print(f"Final accuracy: {metrics.get_accuracy():.2%}")
    print(f"Total questions processed: {metrics.total_questions}")
    print(f"Total correct answers: {metrics.correct_answers}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()