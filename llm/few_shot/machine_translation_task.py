import random
import torch
import numpy as np
from .task_base import TaskBase


def get_language_name(lang_code: str) -> str:
    # simple mapping for common languages
    lang_map = {
        'en': 'English',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'nl': 'Dutch',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'eng_Latn': 'English',
        'swh_Latn': 'Swahili',
        'khm_Khmr': 'Khmer',
    }
    return lang_map[lang_code]


class MachineTranslationTask(TaskBase):

    def __init__(self, dataset_name, chat):
        # name like iwslt2017-en-de
        self.benchname, self.slang, self.tlang = dataset_name.split('-')
        self.slang_name = get_language_name(self.slang)
        self.tlang_name = get_language_name(self.tlang)
        super().__init__(chat=chat, question_prefix=self.slang_name + ':', answer_prefix=self.tlang_name + ':')

    def reset_metric(self):
        self._references = []
        self._hypotheses = []

    def update_metric(self, query_item, pred):
        src = self.get_question(query_item)
        gold = self.get_answer(query_item)
        self._references.append(gold)
        self._hypotheses.append(pred)
        print('=====')
        print(f'SRC: {src}')
        print(f'PRED: {pred}')
        print(f'REF: {gold}')
        print('=====')
        
    def get_metric(self):
        import sacrebleu
        bleu = sacrebleu.metrics.bleu.BLEU()
        score = bleu.corpus_score(self._hypotheses, [self._references], n_bootstrap=10000)
        return {'nsamples': len(self._references), 'bleu': score}

    def get_instruction(self, support_items):
        return f'Please translate {self.slang_name} text into {self.tlang_name}.'
    
    def get_question(self, item):
        if self.benchname == 'iwslt2017':
            return item["translation"][self.slang]
        elif self.benchname == 'flores200':
            return item[f'sentence_{self.slang}']
        else:
            raise ValueError(f'Unknown benchmark name: {self.benchname}')

    def get_answer(self, item):
        if self.benchname == 'iwslt2017':
            return item["translation"][self.tlang]
        elif self.benchname == 'flores200':
            return item[f'sentence_{self.tlang}']
        else:
            raise ValueError(f'Unknown benchmark name: {self.benchname}')

    def sample_few_shot(self, query_item, support_set, num_shot: int):
        indices = random.sample(range(len(support_set)), num_shot)
        return [support_set[i] for i in indices]

    @torch.no_grad()
    def predict(self, tokenizer, model, prompt, device):
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        # generate with beam search
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
        max_new_tokens = tokenizer.model_max_length - inputs.input_ids.shape[1]
        max_new_tokens = min(max_new_tokens, 60)
        if max_new_tokens < 20:
            print(f'Warning: prompt length {inputs.input_ids.shape[1]} is too long.')
        # use beam search with beam size 5
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                                    do_sample=False,
                                    num_beams=5,
                                    num_return_sequences=1,
                                    early_stopping=True,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id,
                                    use_cache=False)
        # decode outputs
        output = outputs[0][inputs.input_ids.shape[1]:]
        pred = tokenizer.decode(output, skip_special_tokens=True)
        pred = pred.strip()
        if pred.find('\n') > 0:
            pred = pred[:pred.find('\n')].strip()
        return pred
