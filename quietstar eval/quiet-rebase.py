import json
import re
import time
import math
import yaml
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List, Dict, Any, Optional, Tuple
import threading
from dataclasses import dataclass
import logging
import os
from datasets import load_dataset
from collections import Counter
import psutil

# Existing configurations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
random_seed = 42
torch.manual_seed(random_seed)
from transformers.models.mistral import configuration_mistral as original_configuration_mistral
from transformers.models.mistral import modeling_mistral as original_modeling_mistral

import configuration_mistral
import modeling_mistral
original_modeling_mistral.MistralModel = modeling_mistral.MistralModel
original_modeling_mistral.MistralForCausalLM = modeling_mistral.MistralForCausalLM
original_configuration_mistral.MistralConfig = configuration_mistral.MistralConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# New Metrics Class
class TestMetrics:
    def __init__(self):
        self.total_questions = 0
        self.correct_answers = 0
        self.predictions = []
        self.ground_truths = []
        self.total_flops = 0
        self.total_time = 0
        
    def update(self, prediction, ground_truth, flops, time_taken):
        self.total_questions += 1  # Increment regardless of prediction

        is_correct = False
        if prediction is not None and ground_truth is not None:
            is_correct = prediction == ground_truth
            self.correct_answers += int(is_correct)
        else:
            self.correct_answers += 0  # Incorrect if prediction or ground truth is None

        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.total_flops += flops
        self.total_time += time_taken
        
    def get_accuracy(self):
        return self.correct_answers / self.total_questions if self.total_questions > 0 else 0
    
    def get_stats(self):
        return {
            "accuracy": self.get_accuracy(),
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "total_flops": self.total_flops,
            "total_time": self.total_time,
            "avg_flops_per_question": self.total_flops / self.total_questions if self.total_questions > 0 else 0,
            "avg_time_per_question": self.total_time / self.total_questions if self.total_questions > 0 else 0
        }

def calculate_flops_per_token(model) -> int:
    """Calculate FLOPs per token using model parameters."""
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_ff = model.config.intermediate_size
    
    N = 2 * d_model * n_layers * (
        2 * (d_model // n_heads) + d_ff
    )
    C_forward = 2 * N + 2 * n_layers * d_model
    return C_forward

def extract_first_integer(s):
    """Extract the first integer from a string."""
    if not s:
        return None
    match = re.search(r'\d+', str(s).replace(',', ''))
    if match:
        return int(match.group())
    return None

@dataclass
class ModelState:
    """State management for model interactions"""
    text: str
    scores: List[float]
    completion_tokens: int
    flops: int = 0
    time_taken: float = 0
    
    def fork(self, n: int) -> List['ModelState']:
        """Create n copies of the current state"""
        return [ModelState(self.text, self.scores.copy(), self.completion_tokens, self.flops, self.time_taken) 
                for _ in range(n)]

class LocalModelManager:
    """Manages QuietStar model for generation within tree search"""
    def __init__(self, config: Dict[str, Any]):
        # Load QuietStar model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config['policy_path'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_thoughts=config.get('n_ahead', 8) + config.get('n_ahead_talk', 1) + 1,
            merged_talk_heads=True,
            merged_lm_and_talk_heads=False,
            merged_lm_and_think_heads=True,
            use_concat_talk_head=True,
            use_shallow_think=True,
            use_shallow_talk=False,
            use_complex_think_head=False,
            use_complex_talk_head=True,
            use_weighted_talk_head=True,
            cache_dir="/scratch/mwa7459/cache"
        )
        
        # Load tokenizer
        self.policy_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.policy_tokenizer.padding_side = "right"
        self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id
        
        # Add special tokens if needed
        special_tokens = []
        if getattr(self.policy_model, 'use_start_thought_token', True):
            special_tokens.append("<|startthought|>")
        if getattr(self.policy_model, 'use_end_thought_token', True):
            special_tokens.append("<|endthought|>")
        if special_tokens:
            self.policy_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.policy_model.resize_token_embeddings(len(self.policy_tokenizer))
        
        # Configure model attributes
        self.policy_model.tokenizer = self.policy_tokenizer
        self.policy_model.gumbel_detach = True
        self.policy_model.include_policy_loss = True
        self.policy_model.use_end_thought_token = True
        self.policy_model.use_start_thought_token = True
        self.policy_model.n_ahead = 8
        self.policy_model.n_ahead_talk = 1
        self.policy_model.n_passes = 1
        self.policy_model.residual_think_head = False
        self.policy_model.optimize_lm_head_only_at_start = False
        self.policy_model.use_policy_loss = False
        self.policy_model.rm_initialized = True
        self.policy_model.first_run = False
        self.policy_model.wandb_enabled = False
        # self.policy_model.config_params = params
        self.policy_model.run_start = int(time.time())
        self.policy_model.eval_mode = True
        self.policy_model.eval()
        
        # Load reward model
        self.reward_tokenizer = AutoTokenizer.from_pretrained(config['reward_path'])
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            config['reward_path'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/scratch/mwa7459/cache"

        )
        self.reward_model.eval()
        
        # Special tokens for reward calculation
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        self.candidate_tokens = self.reward_tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:]
        self.step_tag_id = self.reward_tokenizer.encode(f"{self.step_tag}")[-1]
        
        # Calculate FLOPs
        self.policy_flops_per_token = calculate_flops_per_token(self.policy_model)
        self.reward_flops_per_token = calculate_flops_per_token(self.reward_model)


    def generate_step(self, state: ModelState, max_tokens: int, 
                    temperature: float, stop_str: str) -> ModelState:
        """Generate next step using QuietStar's generation while maintaining ReBase rewards.
        Args:
            state: Current ModelState containing text and scores
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_str: String to stop generation (e.g., "Step X")
        """
        try:
            start_time = time.perf_counter()
            
            # Format prompt for step-by-step reasoning
            current_step_num = len(state.scores) + 1
            if current_step_num == 1:
                formatted_text = f"{state.text}\nA: Let's think step by step."
            else:
                formatted_text = f"{state.text}\nStep {current_step_num}:"

            # Tokenize input
            inputs = self.policy_tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True
            ).to(self.policy_model.device)

            # Initialize generation state
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
            generation_flops = 0
            started_generating_answer_at = None

            # Token-by-token generation with QuietStar's logic
            with torch.no_grad():
                for cur_token_idx in range(max_tokens):
                    # Generate next token logits
                    outputs = self.policy_model(
                        input_ids[~finished_generating],
                        attention_mask=attention_mask[~finished_generating]
                    )
                    new_ids = outputs['logits']
                    new_ids[:, :, self.policy_tokenizer.vocab_size:] = -float("inf")
                    
                    # Track FLOPs
                    generation_flops += self.policy_flops_per_token * (~finished_generating).sum().item()
                    
                    # Process each sequence
                    for list_idx, answer_idx in enumerate((~finished_generating).nonzero(as_tuple=True)[0]):
                        base_answer_ids = input_ids[answer_idx]
                        new_answer_ids = new_ids[list_idx]
                        last_token_idx = (base_answer_ids != self.policy_tokenizer.pad_token_id).nonzero(as_tuple=True)[0].max()
                        
                        # Temperature sampling
                        if temperature == 0:
                            new_ids_sampled = torch.argmax(new_answer_ids[last_token_idx]).unsqueeze(0)
                        else:
                            probs = torch.nn.functional.softmax(new_answer_ids[last_token_idx] / temperature, dim=-1)
                            new_ids_sampled = torch.multinomial(probs, 1)
                        
                        # Extend sequence if needed
                        if last_token_idx + 1 >= len(base_answer_ids):
                            new_padding = torch.full(
                                (len(input_ids), 1), 
                                self.policy_tokenizer.pad_token_id,
                                dtype=torch.long, 
                                device=input_ids.device
                            )
                            input_ids = torch.cat([input_ids, new_padding], dim=-1)
                            attention_mask = torch.cat([attention_mask, torch.zeros_like(new_padding)], dim=-1)
                        
                        attention_mask[answer_idx, last_token_idx + 1] = 1
                        input_ids[answer_idx, last_token_idx + 1] = new_ids_sampled
                        
                        # Check for completion
                        if new_ids_sampled in [self.policy_tokenizer.eos_token_id,
                                            self.policy_tokenizer.bos_token_id,
                                            self.policy_tokenizer.pad_token_id]:
                            finished_generating[answer_idx] = 1
                        
                        # Check and clean up any repeated text
                        decoded = self.policy_tokenizer.decode(input_ids[answer_idx], skip_special_tokens=True)
                        for end_str in ["Q:", "\n\n\n"]:
                            if decoded.count(end_str) > 1:
                                decoded = decoded.split(end_str)[0]
                                new_answer = self.policy_tokenizer.encode(
                                    decoded, 
                                    return_tensors="pt"
                                ).to(input_ids.device)
                                input_ids[answer_idx] = torch.ones_like(input_ids[answer_idx]) * self.policy_tokenizer.pad_token_id
                                input_ids[answer_idx, :new_answer.shape[1]] = new_answer
                                attention_mask[answer_idx] = (input_ids[answer_idx] != self.policy_tokenizer.pad_token_id).long()
                                finished_generating[answer_idx] = 1
                                break
                        
                        # Check for stop string
                        if stop_str in decoded:
                            finished_generating[answer_idx] = 1

                    # Handle final answer generation
                    if cur_token_idx >= 384 and started_generating_answer_at is None:  # QuietStar's start_final_answer_idx
                        started_generating_answer_at = cur_token_idx
                        # Add final answer prompt while preserving previous text
                        base_texts = [self.policy_tokenizer.decode(ids, skip_special_tokens=True) 
                                    for ids in input_ids]
                        final_texts = [text + "\nTherefore, the answer (arabic numerals) is" 
                                    for text in base_texts]
                        encoded_final_texts = self.policy_tokenizer(
                            final_texts, 
                            return_tensors="pt", 
                            padding=True
                        ).to(input_ids.device)
                        attention_mask = encoded_final_texts.attention_mask
                        input_ids = encoded_final_texts.input_ids
                        finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
                    
                    # Check answer length limit
                    if started_generating_answer_at is not None:
                        if cur_token_idx - started_generating_answer_at > 12:  # QuietStar's answer_length
                            break
                    
                    # Break if all finished
                    if finished_generating.all():
                        break

            # Decode final output
            completion = self.policy_tokenizer.decode(
                input_ids[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Handle stop string if present
            if stop_str in completion:
                completion = completion[:completion.index(stop_str)]
            
            # Create new state with reward scoring
            new_text = state.text + completion
            completion_tokens = input_ids.shape[1] - inputs.input_ids.shape[1]
            
            # Calculate reward score (preserve ки token handling)
            step_text = new_text + " ки"
            score = self._calculate_reward(step_text)
            
            # Track metrics
            time_taken = time.perf_counter() - start_time
            total_flops = generation_flops + (state.flops if state else 0)
            
            return ModelState(
                text=new_text,
                scores=state.scores + [score],
                completion_tokens=completion_tokens,
                flops=total_flops,
                time_taken=time_taken + (state.time_taken if state else 0)
            )
            
        except Exception as e:
            logger.error(f"Error in generate_step: {str(e)}")
            return ModelState(state.text, state.scores, 0, state.flops, state.time_taken)
        

    def _calculate_reward(self, text: str) -> float:
        """Calculate reward score using reward model"""
        try:
            if 'ки' not in text:
                text = text.replace('Step', 'Step') + ' ки'

            input_ids = torch.tensor([self.reward_tokenizer.encode(text)]).to(self.reward_model.device)

            with torch.no_grad():
                logits = self.reward_model(input_ids).logits[:,:,self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0]
                step_scores = scores[input_ids == self.step_tag_id]
                return step_scores.mean().item() if len(step_scores) > 0 else 0.0
                
        except Exception as e:
            logging.error(f"Error in _calculate_reward: {str(e)}")
            return 0.0

    def _create_stopping_criteria(self, stop_str: str):
        """Create stopping criteria for text generation"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        tokenized_stop = self.reward_tokenizer.encode(stop_str, add_special_tokens=False)
        
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_token_ids):
                self.stop_token_ids = stop_token_ids

            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_token_ids:
                    if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                        return True
                return False
            
            def __len__(self):
                return len(self.stop_token_ids)

        return StoppingCriteriaList([StopOnTokens([tokenized_stop])])

@dataclass
class ModelState:
    """State management for tree search"""
    text: str
    scores: List[float]
    completion_tokens: int
    flops: int = 0
    time_taken: float = 0
    
    def fork(self, n: int) -> List['ModelState']:
        """Create n copies of the current state"""
        return [ModelState(self.text, self.scores.copy(), self.completion_tokens, 
                         self.flops, self.time_taken) for _ in range(n)]

class TreeNode:
    """Node in the search tree representing a state in the solution process"""
    def __init__(self, id: int, state: ModelState, score: float, 
                 num_step_tokens: int = 0, parent: Optional['TreeNode'] = None):
        self.id = id
        self.state = state
        self.text_ = state.text
        self.score_ = score
        self.parent = parent
        self.leaf_ = False
        self.cum_tokens = 0
        self.num_step_tokens = num_step_tokens
        
        # Check for answer completion
        if parent is not None and self._is_answer_complete():
            self.leaf_ = True
            
        # Update depth and token count
        if parent is not None:
            self.depth = parent.get_depth() + 1
            self.cum_tokens = parent.get_cum_tokens() + num_step_tokens
        else:
            self.depth = 0
            self.cum_tokens = num_step_tokens
    def _is_answer_complete(self) -> bool:
        """
        Determine if the node's text indicates that an answer has been provided.
        """
        answer_indicators = [
            "The answer is",
            "Therefore,",
            "Thus,",
            "In conclusion,",
            "Answer:",
            "Hence,",
            "####",  # GSM8K format
            "So, the number of",  # Add more patterns as needed
        ]
        # Check if any of the indicators are in the text
        if any(indicator in self.text_ for indicator in answer_indicators):
            return True

        # Alternatively, check if the final section contains numbers
        final_section = find_final_section(self.text_)
        numbers = re.findall(r'\b\d+\b', final_section)
        if numbers:
            return True

    def get_id(self): return self.id
    def get_parent(self): return self.parent
    def get_text(self): return self.text_
    def get_state(self): return self.state
    def get_depth(self): return self.depth
    def get_score(self): return self.score_
    def is_leaf(self): return self.leaf_
    def get_cum_tokens(self): return self.cum_tokens

class Tree:
    """Search tree implementation using QuietStar for generation"""
    def __init__(self, root_state: ModelState, paras: Dict[str, Any], model_manager: LocalModelManager):
        self.size_ = 1
        self.nodes = []
        self.paras = paras
        self.model_manager = model_manager
        self.root_ = TreeNode(0, root_state, 1.0)
        self.remaining_width = paras["width"]
        self.history_list = []
        self.running_list = []
        self.depth_nodes = [[] for _ in range(100)]
        self.nodes.append(self.root_)
        self.depth_nodes[0].append(self.root_)
    
    def reset_running_list(self):
        self.running_list = []
    
    def get_running_list(self): return self.running_list
    def get_history_list(self): return self.history_list
    def get_nodes(self): return self.nodes
    
    def expand(self, node: TreeNode, wid: int):
        """Expand a node using QuietStar generation"""
        state = node.get_state()
        forks = state.fork(wid)
        depth = node.get_depth()
        
        for fork in forks:
            new_state = self.model_manager.generate_step(
                fork,
                self.paras["max_step_tokens"],
                self.paras["temperature"],
                f"Step {depth+2}"
            )
            
            # Only add states with valid scores
            if new_state.scores and new_state.scores[-1] > 0.1:
                self.running_list.append((new_state, node))
                self.history_list.append(new_state)
    
    def insert(self, state: ModelState, parent: TreeNode):
        """Insert a new node into the tree"""
        if not state.scores:
            return
        score = state.scores[-1]
        new_node = TreeNode(self.size_, state, score, state.completion_tokens, parent)
        self.size_ += 1
        depth = new_node.get_depth()
        self.depth_nodes[depth].append(new_node)
        self.nodes.append(new_node)

    def select_softmax(self, node_list, node_weights, width):
        """Select nodes using softmax distribution"""
        if not node_list:
            return [], []
            
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_pairs = sorted(node_weight_pair_list, key=lambda pair: pair[1], reverse=True)
        
        nodes = [pair[0] for pair in sorted_pairs]
        weights = torch.tensor([pair[1] for pair in sorted_pairs])
        
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        
        select_num = []
        remaining_width = width
        remaining_sum = sum_exp_weights.clone()
        
        for weight in exp_weights:
            if remaining_sum <= 0 or remaining_width <= 0:
                select_num.append(0)
                continue
                
            num = int(math.ceil(remaining_width * weight / remaining_sum))
            select_num.append(num)
            remaining_width -= num
            remaining_sum -= weight
            
        return nodes, select_num

    def select_softmax_with_truncation(self, node_list, node_weights, width):
        """Select nodes using softmax with truncation"""
        if not node_list:
            return [], []
            
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_pairs = sorted(node_weight_pair_list, key=lambda pair: pair[1], reverse=True)
        
        truncate_ratio = self.paras["truncate_ratio"]
        keep_num = max(1, int(math.ceil(len(sorted_pairs) * truncate_ratio)))
        truncated_pairs = sorted_pairs[:keep_num]
        
        nodes = [pair[0] for pair in truncated_pairs]
        weights = torch.tensor([pair[1] for pair in truncated_pairs])
        
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        
        select_num = []
        remaining_width = width
        remaining_sum = sum_exp_weights.clone()
        
        for weight in exp_weights:
            if remaining_sum <= 0 or remaining_width <= 0:
                select_num.append(0)
                continue
                
            num = int(math.ceil(remaining_width * weight / remaining_sum))
            select_num.append(num)
            remaining_width -= num
            remaining_sum -= weight
            
        return nodes, select_num

    def select_and_expand(self, depth):
        """Select nodes to expand and perform expansion"""
        cand_node_list = []
        cand_node_weights = []
        
        for node in self.depth_nodes[depth]:
            if node.is_leaf() or node.get_cum_tokens() >= self.paras["max_tokens"]:
                self.remaining_width -= 1
            else:
                cand_node_list.append(node)
                cand_node_weights.append(node.get_score())
                
        if self.remaining_width <= 0 or not cand_node_list:
            return False
            
        if self.paras["select_method"] == "softmax":
            nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, self.remaining_width)
        elif self.paras["select_method"] == "softmax_with_truncate":
            nodes, widths = self.select_softmax_with_truncation(cand_node_list, cand_node_weights, self.remaining_width)
        else:
            logger.error(f"Unknown selection method: {self.paras['select_method']}")
            return False
            
        for expand_node, width in zip(nodes, widths):
            if width >= 1:
                self.expand(expand_node, width)
        return True


def find_final_section(text: str) -> str:
    """
    Intelligently identify the final section containing the answer by analyzing
    the structure and context of the text, rather than relying on hardcoded markers.
    """
    # First check for GSM8K standard format
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # Split text into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return text
        
    # Look at the last few sentences for numerical content
    last_sentences = sentences[-3:]  # Consider last 3 sentences for context
    for sentence in reversed(last_sentences):
        if re.search(r'\d', sentence):  # If sentence contains any numbers
            return sentence
            
    # If no numbers found in last sentences, return last sentence
    return sentences[-1]

def extract_final_answer(text: str) -> Optional[int]:
    """
    Extract the final answer by analyzing the mathematical context and progression
    of numbers in the text, rather than relying on specific markers.
    """
    # Get the final section of text most likely to contain the answer
    final_section = find_final_section(text)
    
    # Find all numbers in the final section
    numbers = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)', final_section)
    
    if not numbers:
        # If no numbers in final section, look through whole text
        numbers = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)', text)
        
    if numbers:
        # Convert the last number found
        return int(numbers[-1].replace(',', ''))
        
    return None

def extract_ground_truth(ground_truth: str) -> Optional[int]:
    """
    Extract the ground truth answer from the text, handling GSM8K and other formats.
    """
    # Check for GSM8K format first
    if "####" in ground_truth:
        final_section = ground_truth.split("####")[-1].strip()
        numbers = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)', final_section)
        if numbers:
            return int(numbers[-1].replace(',', ''))
    
    # If not in GSM8K format or no number found, find all numbers
    numbers = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)', ground_truth)
    if numbers:
        return int(numbers[-1].replace(',', ''))
    
    return None

def validate_answer(prediction: Optional[int], ground_truth: str) -> bool:
    """Validate model prediction against ground truth."""
    if prediction is None:
        return False
    
    ground_truth_number = extract_ground_truth(ground_truth)
    if ground_truth_number is None:
        return False
    
    return prediction == ground_truth_number

def process_answers(node_text: str) -> Optional[Dict[str, Any]]:
    """Process and extract answers from node text with intelligent section analysis."""
    final_section = find_final_section(node_text)
    answer = extract_final_answer(node_text)
    
    if answer is not None:
        return {
            "answer": answer,
            "text": node_text,
            "final_section": final_section
        }
    return None
    
def reward_guided_search(question: str, id: int, ground_truth_answer: Dict[str, Any], 
                        paras: Dict[str, Any], model_manager: LocalModelManager,
                        metrics: TestMetrics) -> Dict[str, Any]:
    """Perform reward-guided search using local models with detailed response logging"""
    try:
        start_time = time.perf_counter()
        initial_state = ModelState(question, [], 0)
        tree = Tree(initial_state, paras, model_manager)
        depth = 0
        total_flops = 0
        
        # Create debug log file for this question
        debug_path = f"{paras['store_path']}/debug_q{id}"
        os.makedirs(debug_path, exist_ok=True)
        debug_file = f"{debug_path}/model_responses.txt"
        
        with open(debug_file, "w", encoding='utf-8') as f:
            f.write(f"Question {id}:\n{question}\n")
            f.write("\nGround Truth:\n{}\n".format(ground_truth_answer.get('answer', '')))
            f.write("\n" + "="*50 + "\n")
        
        while True:
            tree.reset_running_list()
            continue_search = tree.select_and_expand(depth)
            if not continue_search:
                break
                
            running_list = tree.get_running_list()
            for state, parent in running_list:
                tree.insert(state, parent)
                total_flops += state.flops
                
                # Log each generation step
                with open(debug_file, "a", encoding='utf-8') as f:
                    f.write(f"\nDepth {depth}, Node {tree.size_ - 1}:\n")
                    f.write("Full Response:\n")
                    f.write(state.text)
                    f.write("\n")
                    f.write(f"Scores: {state.scores}\n")
                    f.write("-"*50 + "\n")
            
            depth += 1
            if depth >= 25:  # Maximum depth limit
                break

        # Process results
        total_time = time.perf_counter() - start_time
        history_list = tree.get_history_list()
        total_tokens = sum(state.completion_tokens for state in history_list)

        # Extract and log answers
        answers = []
        with open(debug_file, "a", encoding='utf-8') as f:
            f.write("\nFinal Answers:\n")
            
        for node in tree.get_nodes():
            if node.is_leaf():
                text = node.get_text()
                extracted = process_answers(text)
                if extracted:
                    answer_info = {
                        "answer": extracted["answer"],
                        "score": node.get_score(),
                        "step_scores": node.state.scores,
                        "text": extracted["text"],
                        "final_section": extracted["final_section"]
                    }
                    answers.append(answer_info)
                    
                    # Log each answer
                    with open(debug_file, "a", encoding='utf-8') as f:
                        f.write(f"\nAnswer: {answer_info['answer']}\n")
                        f.write(f"Score: {answer_info['score']}\n")
                        f.write(f"Step Scores: {answer_info['step_scores']}\n")
                        f.write(f"Full Text:\n{answer_info['text']}\n")
                        f.write(f"Final Section: {answer_info['final_section']}\n")
                        f.write("-"*50 + "\n")

        # Sort answers by score and get best answer
        answers.sort(key=lambda x: x["score"], reverse=True)
        best_answer = answers[0]["answer"] if answers else None
        
        # Update metrics
        true_answer = extract_ground_truth(ground_truth_answer["answer"])
        metrics.update(best_answer, true_answer, total_flops, total_time)

        # Log final results
        with open(debug_file, "a", encoding='utf-8') as f:
            f.write("\nFinal Results:\n")
            f.write(f"Best Answer: {best_answer}\n")
            f.write(f"Ground Truth: {true_answer}\n")
            f.write(f"Correct: {best_answer == true_answer if best_answer is not None else False}\n")
            f.write(f"Total Tokens: {total_tokens}\n")
            f.write(f"Total FLOPs: {total_flops}\n")
            f.write(f"Time Taken: {total_time:.2f}s\n")

        result = {
            "id": id,
            "question": question,
            "model_answer": answers,
            "ground_truth_answer": ground_truth_answer["answer"],
            "total_tokens": total_tokens,
            "total_flops": total_flops,
            "time_taken": total_time,
            "correct": best_answer == true_answer if best_answer is not None else False,
            "metrics": metrics.get_stats(),
            "debug_file": debug_file  # Include path to debug file in results
        }

        # Save results
        result_path = f"{paras['store_path']}/answer_q{id}.json"
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return result

    except Exception as e:
        logger.error(f"Error in reward_guided_search: {str(e)}")
        return {
            "id": id,
            "question": question,
            "model_answer": [],
            "ground_truth_answer": ground_truth_answer["answer"],
            "total_tokens": 0,
            "total_flops": 0,
            "time_taken": 0,
            "error": str(e)
        }

def process_gsm8k_dataset(config: Dict[str, Any], split: str = "test") -> List[Dict[str, Any]]:
    """Process questions from the GSM8K dataset using the reward-guided search approach."""
    try:
        # Load GSM8K dataset
        logger.info(f"Loading GSM8K dataset ({split} split)")
        dataset = load_dataset("gsm8k", "main", split=split).shuffle(seed=random_seed)
        
        # Load parameters from yaml
        logger.info(f"Loading parameters from {config['parameter_path']}")
        with open(config['parameter_path'], 'r', encoding='utf-8') as f:
            paras = yaml.safe_load(f)
        
        # Ensure store path exists
        os.makedirs(paras['store_path'], exist_ok=True)
        
        # Initialize model manager and metrics
        model_manager = LocalModelManager(config)
        metrics = TestMetrics()
        
        # Process questions
        results = []
        batch_size = config['batch_size']
        
        for i in range(0, 128, batch_size):
            batch_indices = list(range(i, min(i + batch_size, 128)))
            logger.info(f"Processing batch {i//batch_size + 1}/{math.ceil(128/batch_size)}")
            
            for j, idx in enumerate(batch_indices):
                try:
                    item = dataset[idx]
                    question = item['question']
                    ground_truth = {
                        "answer": item['answer'],
                        "solution": item.get('solution', '')
                    }
                    
                    result = reward_guided_search(
                        question=question,
                        id=i + j,
                        ground_truth_answer=ground_truth,
                        paras=paras,
                        model_manager=model_manager,
                        metrics=metrics
                    )
                    results.append(result)
                    
                    # Log progress
                    logger.info(f"Question {i + j} completed:")
                    logger.info(f"Time taken: {result['time_taken']:.2f}s")
                    logger.info(f"Total FLOPs: {result['total_flops']:,}")
                    logger.info(f"Current accuracy: {metrics.get_accuracy():.2%}")
                    
                    
                except Exception as e:
                    logger.error(f"Error processing question {i + j}: {str(e)}")
                    continue
                
            # Save intermediate results
            try:
                with open(config['output_path'], "w", encoding='utf-8') as f:
                    json.dump({
                        "results": results,
                        "metrics": metrics.get_stats()
                    }, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error saving intermediate results: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in process_gsm8k_dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    config = {
        "output_path": "PATH_TO_SAVE_RESULTS.json",
        "parameter_path": "rebase.yaml",
        "policy_path": "ezelikman/quietstar-8-ahead",
        "reward_path": "peiyi9979/math-shepherd-mistral-7b-prm",
        "batch_size": 1
    }
    
    try:
        # Initialize metrics tracking file
        metrics_file = f"{config['output_path']}_metrics.txt"
        with open(metrics_file, "w", encoding='utf-8') as f:
            f.write("=== GSM8K Evaluation Metrics ===\n\n")
        
        # Process the GSM8K dataset
        results = process_gsm8k_dataset(config)
        
        # Save final results with metrics
        final_metrics = {
            "results": results,
            "summary": {
                "total_questions": len(results),
                "total_correct": sum(1 for r in results if r.get("correct", False)),
                "total_flops": sum(r.get("total_flops", 0) for r in results),
                "total_time": sum(r.get("time_taken", 0) for r in results),
                "average_flops_per_question": sum(r.get("total_flops", 0) for r in results) / len(results) if results else 0,
                "average_time_per_question": sum(r.get("time_taken", 0) for r in results) / len(results) if results else 0,
                "accuracy": sum(1 for r in results if r.get("correct", False)) / len(results) if results else 0
            }
        }
        
        with open(config["output_path"], "w", encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=4, ensure_ascii=False)
            
        # Save detailed metrics to text file
        with open(metrics_file, "a", encoding='utf-8') as f:
            f.write("\n=== Final Summary ===\n")
            f.write(f"Total Questions: {final_metrics['summary']['total_questions']}\n")
            f.write(f"Total Correct: {final_metrics['summary']['total_correct']}\n")
            f.write(f"Accuracy: {final_metrics['summary']['accuracy']:.2%}\n")
            f.write(f"Total FLOPs: {final_metrics['summary']['total_flops']:,}\n")
            f.write(f"Total Time: {final_metrics['summary']['total_time']:.2f} seconds\n")
            f.write(f"Average FLOPs per Question: {final_metrics['summary']['average_flops_per_question']:,.2f}\n")
            f.write(f"Average Time per Question: {final_metrics['summary']['average_time_per_question']:.2f} seconds\n")
        logger.info("Successfully completed processing GSM8K dataset")
        logger.info(f"Final Accuracy: {final_metrics['summary']['accuracy']:.2%}")
        logger.info(f"Total FLOPs: {final_metrics['summary']['total_flops']:,}")
        logger.info(f"Total Time: {final_metrics['summary']['total_time']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
