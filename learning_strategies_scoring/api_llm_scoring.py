import torch
import json
from os import path
from vllm import LLM
from vllm.sampling_params import SamplingParams
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

# Essential NCCL settings for network compatibility
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_NET_GDR_LEVEL'] = '0'

class LLMScoring:
    def __init__(self, model_path):
        """
        model_path: str
            Path to the model to be used for scoring
        """

        self.model = LLM(
            model=model_path,
            dtype=torch.bfloat16,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=2,      # Enable both GPUs
            enforce_eager=True,
            enable_prefix_caching=False,
            disable_custom_all_reduce=True
        )
        self.scoring_details_dir = path.join('learning_strategies_scoring', 'scoring_details')
        self.params = SamplingParams(temperature=0, max_tokens=300)
        self._initialize_score_tokens()

    def generate_response(self, formatted_prompt):
        """
        formatted_prompt: str
            Prompt to be used for generating the response

        Returns:
            str: Generated response
        """
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.model.chat(messages, self.params)

        return response[0].outputs[0].text

    def extract_score_from_response(self, response):
        """
        response: str
            Response from the model

        Returns:
            dict: Dictionary containing the scores
        """
        lines = response.split('\n')

        scores = {}
        for line in lines:
            if line.startswith('- '):
                # Remove <|endoftext|> from the line if it exists
                line = line.replace('<|endoftext|>', '')
                line = line.replace('<|im_start|>', '')

                try:
                    key, value = line.split(': ')
                    if key[2:] not in scores:
                        scores[key[2:]] = value
                except:
                    pass
        return scores

    def prepare_scoring_rubric_prompt(self, scoring_details):
        """
        scoring_details: dict
            Scoring details dictionary

        Returns:
            str: Scoring rubric prompt
        """

        task_prompt = scoring_details['task']
        scoring_rubric_prompt = ""
        for _, dict in scoring_details['scoring_rubric'].items():
            descriptions = '\n'.join([f"- - {dict['scores'][num]}: {dict['scores_description'][num]}" for num in dict['scores']])
            scoring_rubric_prompt += f"- {dict['name']}:\n{descriptions}\n"
        scoring_rubric_prompt = scoring_rubric_prompt[:-1]

        return task_prompt, scoring_rubric_prompt


    def prepare_prompt(self, data, task):
        """
        data: dict
            Data to be used for scoring
        task: str
            Task to be scored

        Returns:
            str: Formatted prompt
        """

        scoring_start_prompt = "Rate the quality of the following performed task, based on the scoring rubric."

        if task == 'selfexplanation':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence, and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_se.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'thinkaloud':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence, and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_ta.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'summary':
            if 'context' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'summaries_aloe.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'paraphrasing':
            if 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain target_sentence and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'paraphrasing_ulpc.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Sentence: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        return prompt

    def score(self, data, task):
        """
        data: dict
            Data to be used for scoring
        task: str
            Task to be scored

        Returns:
            dict: Dictionary containing the scores
        """

        formatted_prompt = self.prepare_prompt(data, task)
        response = self.generate_response(formatted_prompt)
        scores = self.extract_score_from_response(response)

        return scores

    def _initialize_score_tokens(self):
        """
        Initialize the exact token IDs from tokenizer.json
        """
        self.score_token_ids = {
            "Poor": [85203, 45773],      # "Poor", "ĠPoor" 
            "Fair": [61895, 14930],      # "Fair", "ĠFair"
            "Good": [15571, 7839],       # "Good", "ĠGood"  
            "Excellent": [50755, 37866]  # "Excellent", "ĠExcellent"
        }
        self.score_labels = ["Poor", "Fair", "Good", "Excellent"]

    def generate_response_with_logprobs(self, formatted_prompt, logprobs=10):
        """
        Generate response with logprobs for confidence calculation.
        """
        from vllm import SamplingParams
        
        params = SamplingParams(
            temperature=0, 
            max_tokens=300,
            logprobs=logprobs
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        outputs = self.model.chat(messages, params)
        output = outputs[0].outputs[0]
        
        return {
            'text': output.text,
            'token_logprobs': output.logprobs
        }

    def extract_dimension_confidences(self, response_data):
        """
        Extract confidence scores for each dimension from logprobs.
        """
        text = response_data['text']
        token_logprobs = response_data['token_logprobs']
        
        # Parse response to find dimension-score pairs (skip "Not attempted")
        dimension_scores = {}
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
                
            match = re.match(r'-\s*([^:]+):\s*(.+)', line)
            if match:
                dimension = match.group(1).strip()
                score = match.group(2).strip()
                if score in self.score_labels:  # Only track Poor, Fair, Good, Excellent
                    dimension_scores[dimension] = score
        
        # Calculate confidence for each dimension
        confidences = {}
        for dimension, predicted_score in dimension_scores.items():
            confidence = self._calculate_score_confidence(token_logprobs, predicted_score)
            confidences[dimension] = confidence
            
        return confidences

    def _calculate_score_confidence(self, token_logprobs, predicted_score):
        """
        Calculate confidence for a specific score prediction.
        Returns the confidence of the chosen score + probabilities of all scores.
        """
        if not hasattr(self, 'score_token_ids'):
            self._initialize_score_tokens()
        
        # Find position where this score appears
        score_position = self._find_score_position(token_logprobs, predicted_score)
        
        if score_position is None:
            error_response = {
                'confidence': 0.0,
                'predicted_score': predicted_score,
                'score_probabilities': {label: 0.0 for label in self.score_labels},
                'error': 'Could not find score token position'
            }
            
            # Include debug info if available
            if hasattr(self, '_find_debug') and predicted_score in self._find_debug:
                error_response['debug'] = self._find_debug[predicted_score]
            
            return error_response
        
        # Get probabilities for all four score labels at this position
        logprobs_dict = token_logprobs[score_position]
        score_probs = {}
        
        for label in self.score_labels:
            prob = self._get_probability_for_label(logprobs_dict, label)
            score_probs[label] = prob
        
        # Calculate normalized confidence
        total_prob = sum(score_probs.values())
        predicted_prob = score_probs.get(predicted_score, 0.0)
        confidence = predicted_prob / total_prob if total_prob > 0 else 0.0
        
        return {
            'confidence': float(confidence),
            'predicted_score': predicted_score,
            'score_probabilities': {k: float(v) for k, v in score_probs.items()},
            'total_probability_mass': float(total_prob)
        }

    def _find_score_position(self, token_logprobs, predicted_score):
        """
        Find where the predicted score appears in the token sequence.
        We need to find where this score token was actually CHOSEN (rank=1).
        """
        target_token_ids = set(self.score_token_ids[predicted_score])
        
        for i, logprobs_dict in enumerate(token_logprobs):
            if isinstance(logprobs_dict, dict):
                # Find which token was actually chosen (rank=1)
                for token_id, logprob_obj in logprobs_dict.items():
                    if (hasattr(logprob_obj, 'rank') and logprob_obj.rank == 1 and 
                        token_id in target_token_ids):
                        return i
        
        return None

    def _get_probability_for_label(self, logprobs_dict, label):
        """
        Get probability for a score label from token logprobs dict.
        logprobs_dict has token_ids as keys and Logprob objects as values.
        """
        target_token_ids = set(self.score_token_ids[label])
        total_prob = 0.0
        
        if isinstance(logprobs_dict, dict):
            for token_id, logprob_obj in logprobs_dict.items():
                if token_id in target_token_ids:
                    # Extract the actual logprob value from the Logprob object
                    if hasattr(logprob_obj, 'logprob'):
                        total_prob += np.exp(logprob_obj.logprob)
                    elif hasattr(logprob_obj, 'log_prob'): 
                        total_prob += np.exp(logprob_obj.log_prob)
        
        return total_prob

    def score_with_confidence(self, data, task='summary'):
        """
        Get both scores and confidence metrics.
        """
        if not hasattr(self, 'score_token_ids'):
            self._initialize_score_tokens()
        
        formatted_prompt = self.prepare_prompt(data, task)
        response_data = self.generate_response_with_logprobs(formatted_prompt)
        
        # Extract all scores (including "Not attempted")
        all_scores = self.extract_score_from_response(response_data['text'])
        
        # Extract confidence only for trackable scores (Poor, Fair, Good, Excellent)
        confidences = self.extract_dimension_confidences(response_data)
        
        result = {
            'scores': all_scores,
            'confidences': confidences,
            'raw_response': response_data['text']
        }
        
        # Include debug info if available
        if hasattr(self, '_debug_info'):
            result['debug_info'] = self._debug_info
            delattr(self, '_debug_info')  # Clean up
        
        return result
