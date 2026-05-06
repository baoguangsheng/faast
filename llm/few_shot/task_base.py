import random
import torch
import numpy as np
from transformers import AutoTokenizer


class TaskBase:

    def __init__(self, chat: bool, question_prefix: str, answer_prefix: str):
        self.chat = chat
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.reset_metric()

    def reset_metric(self):
        raise NotImplementedError

    def update_metric(self, query_item, prediction):
        raise NotImplementedError

    def get_metric(self):
        raise NotImplementedError
    
    def sample_few_shot(self, query_item, support_set, num_shot: int):
        # default random sampling
        indices = random.sample(range(len(support_set)), num_shot)
        return [support_set[i] for i in indices]

    def get_systemprompt(self):
        return "You are a helpful assistant."
    
    def get_instruction(self, support_items):
        raise NotImplementedError
    
    def get_question(self, item: dict):
        raise NotImplementedError
    
    def get_answer(self, item: dict):
        raise NotImplementedError

    def _format_chat(self, tokenizer: AutoTokenizer, prompt: str, response: str):
        messages = [{'role': 'system', 'content': self.get_systemprompt()},
                   {'role': 'user', 'content': prompt}]
        if response is None:
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            messages.append({'role': 'assistant', 'content': response})
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        return prompt_text
    
    def build_prompt(self, tokenizer: AutoTokenizer, query_item: dict = None, support_items: list = [], include_response: bool = False):
        # format ICL-style prompt
        prompts = []
        for item in support_items:
            prompt = f"{self.question_prefix} {self.get_question(item)}\n"
            prompt += f"{self.answer_prefix} {self.get_answer(item)}\n"
            prompts.append(prompt)
        if query_item is not None:
            prompt = f"{self.question_prefix} {self.get_question(query_item)}\n"
            prompt += f"{self.answer_prefix}"
            prompts.append(prompt)
        prompt = '\n'.join(prompts)
        # add instruction if any
        if self.chat:
            instruction = self.get_instruction(support_items)
            response = self.get_answer(query_item) if include_response else None
            return self._format_chat(tokenizer, instruction + '\n\n' + prompt, response)
        else:
            if include_response:
                prompt += f" {self.get_answer(query_item)}\n"
            return prompt

    def predict(self, tokenizer, model, prompt, device):
        raise NotImplementedError

