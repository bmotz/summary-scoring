import torch
import json
from os import path
from vllm import LLM
from vllm.sampling_params import SamplingParams
import os

# Force vLLM V0 engine
os.environ['VLLM_USE_V1'] = '0'

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
            gpu_memory_utilization=0.4,
            tensor_parallel_size=2,      # Enable both GPUs
            enforce_eager=True, 
            enable_prefix_caching=False, 
            disable_custom_all_reduce=True
        )
        self.scoring_details_dir = path.join('learning_strategies_scoring', 'scoring_details')
        self.params = SamplingParams(temperature=0, max_tokens=300)
    
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
