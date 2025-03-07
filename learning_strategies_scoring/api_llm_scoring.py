import torch
import torch.backends.cpu
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from os import path

class LLMScoring:
    def __init__(self, model_path, device):
        """
        model_path: str
            Path to the model to be used for scoring
        device: str
            Device to run the model on. 'cuda' for GPU, 'cpu' for CPU, 'mps' for Mac GPU
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set thread count for CPU optimization
        torch.backends.cpu.num_threads = 8
        torch.set_num_threads(8)

        # Add CPU optimization configurations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map='cpu'
        )
        self.device = device
        self.model.to(self.device)
        self.scoring_details_dir = path.join('learning_strategies_scoring', 'scoring_details')
    
    def generate_response(self, formatted_prompt):
        """
        formatted_prompt: str
            Prompt to be used for generating the response

        Returns:
            str: Generated response
        """
        model_inputs = self.tokenizer(
            [formatted_prompt], 
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            max_length=1024 # Reduced from 2048*2
        ).to(self.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=100, # Reduced from 300
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            top_p=None,
            top_k=None,
            temperature=None,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response[0]
    
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
    
        scoring_rubric_prompt = ""
        for _, dict in scoring_details['scoring_rubric'].items():
            descriptions = '\n'.join([f"- - {dict['scores'][num]}: {dict['scores_description'][num]}" for num in dict['scores']])
            scoring_rubric_prompt += f"- {dict['name']}:\n{descriptions}\n"
        scoring_rubric_prompt = scoring_rubric_prompt[:-1]

        return scoring_rubric_prompt


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
                raise ValueError('Data must contain context, target_sentence and student_response fields')
            
            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_se.json'), 'r'))
            task_prompt = scoring_details['task']
            scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'thinkaloud':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_ta.json'), 'r'))
            task_prompt = scoring_details['task']
            scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'summary':
            if 'context' not in data or 'question' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, question and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'summaries_classe.json'), 'r'))
            scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            task_prompt = data['question']

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        
        elif task == 'paraphrasing':
            if 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain target_sentence and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'paraphrasing_ulpc.json'), 'r'))
            task_prompt = scoring_details['task']
            scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Sentence: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        return f"{prompt}\n\n### Response:"

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