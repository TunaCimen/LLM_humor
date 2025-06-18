import re
import json
import difflib
import os

import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import utils
from retrieval import MedRAG
from tqdm import tqdm
import re
import difflib
import os

        
class InformationGainScorer:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def calculate_perplexity(self, logits, target, attention_mask):
        """
        Calculate perplexity from logits and target labels.

        Args:
        - logits (torch.Tensor): Logits output from the model (batch_size, seq_length, vocab_size).
        - target (torch.Tensor): Ground truth labels (batch_size, seq_length).

        Returns:
        - perplexity (float): The perplexity score.
        """

        shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
        shift_labels = target[:, 1:] # shift by 1 token position

        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather the log probabilities for the correct tokens
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask out positions corresponding to padding tokens
        target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

        # Compute the mean negative log-likelihood for each sequence
        negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)

        perplexities = torch.exp(negative_log_likelihood)

        return perplexities

    def calculate_true_answer_probability(self, steps_text, true_answer):
        """
        Given the current steps in text format, calculate the probability of the true answer.
        
        Args:
        - steps_text: A string representing the steps taken so far.
        - true_answer: The correct answer in text.

        Returns:
        - Probability of the true answer given the steps.
        """
        # Tokenize input (steps) and true answer
        

        inputs = self.tokenizer(steps_text + ' Answer: ' + true_answer, return_tensors="pt")
        input_with_answer_ids = inputs.input_ids
        attention_mask_with_answer_ids = inputs.attention_mask

        with torch.no_grad():
            logits = self.model(input_with_answer_ids.to(self.model.device), return_dict=True).logits.cpu()
        answer_perplexity = self.calculate_perplexity(logits, input_with_answer_ids, attention_mask_with_answer_ids)

        return answer_perplexity.item()

 
    def information_gain_score_step(self, cumulative_step_knowledge: str, answer: str):
        """
        Args:
            cumulative_step_knowledge (str): knowledge text for reasoning step so far
            answer (str): true answer
        Returns:
            information_gain_score (float): retrieval score for the reasoning step
        """
        true_answer_probability = self.calculate_true_answer_probability(cumulative_step_knowledge, answer)
        return true_answer_probability
        
    
    def forward(self, question, reasoning_steps, answer):
        """
        Args:
            reasoning_steps (dict): dictionary containing the reasoning steps
        Returns:
            retrieval_scores (list): list of retrieval scores for each reasoning step
        """
        information_gain_scores = {}
        information_gain_scores['Steps'] = []
        information_gain_scores['Mean'] = 0
        information_gain_scores['Info_gain'] = 0

        if len(reasoning_steps['steps']) <= 2:
            return information_gain_scores
            
        cumulative_step_actions = question

        information_gain_score = self.information_gain_score_step(cumulative_step_actions, answer)
        information_gain_scores['Steps'].append(information_gain_score)
        
        for step in reasoning_steps['steps']:
            cumulative_step_actions += '\n' + step['knowledge']
            information_gain_score = self.information_gain_score_step(cumulative_step_actions, answer)
            information_gain_scores['Steps'].append(information_gain_score)
            diff = information_gain_scores['Steps'][-2] - information_gain_scores['Steps'][-1]
            information_gain_scores['Info_gain'] += diff
            information_gain_scores['Mean'] += information_gain_score
        information_gain_scores['Mean'] /= len(information_gain_scores['Steps'])
        information_gain_scores['Info_gain'] /= len(information_gain_scores['Steps'])
        
        return information_gain_scores    