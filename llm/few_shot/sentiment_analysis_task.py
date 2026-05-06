import random
import torch
import numpy as np
from .task_base import TaskBase


class SentimentAnalysisTask(TaskBase):

    def __init__(self, dataset_name, chat):
        assert chat == False, "SentimentAnalysisTask does not support chat mode."
        super().__init__(chat=False, question_prefix='Review:', answer_prefix='Sentiment:')
        self.label_map = {0: "negative", 1: "positive"}
        self.max_words = 100 if dataset_name == 'sst2' else 300

    def reset_metric(self):
        self._total = 0
        self._correct = 0

    def update_metric(self, query_item, pred):
        gold = self.get_answer(query_item)
        self._total += 1
        if pred == gold:
            self._correct += 1

    def get_metric(self):
        acc = self._correct / self._total if self._total > 0 else 0.0
        ci95 = 1.96 * np.sqrt(acc * (1 - acc) / self._total) if self._total > 0 else 0.0
        return {'nsamples': self._total, 'acc': round(acc, 4), 'ci95': round(float(ci95), 4)}

    def _truncate(self, text):
        max_words = self.max_words
        words = text.split(' ')
        if len(words) >= max_words:
            print(f'Warning: truncating text from {len(words)} to {max_words} words.')
            words = words[:max_words]
            return ' '.join(words)
        return text

    def get_question(self, item):
        if "sentence" in item:
            return self._truncate(item["sentence"])
        elif "text" in item:
            return self._truncate(item["text"])
        else:
            raise ValueError("Item does not contain 'sentence' or 'text' field.")

    def get_answer(self, item):
        return self.label_map[item["label"]]
    
    def sample_few_shot(self, query_item, support_set, num_shot: int):
        def _get_label_indices(dataset):
            if hasattr(dataset, 'label2indices'):
                return dataset.label2indices
            # build label to indices mapping
            label2indices = {}
            for idx, item in enumerate(dataset):
                label = self.get_answer(item)
                if label not in label2indices:
                    label2indices[label] = []
                label2indices[label].append(idx)
            setattr(dataset, 'label2indices', label2indices)
            return label2indices
        
        # sample num_shot samples for each label
        label2indices = _get_label_indices(support_set)
        indices = []
        for label in label2indices:
            if len(label2indices[label]) < num_shot:
                raise ValueError(f"Not enough samples for label {label} to sample {num_shot} shots.")
            indices.extend(random.sample(label2indices[label], num_shot))
        random.shuffle(indices)
        return [support_set[i] for i in indices]

    @torch.no_grad()
    def predict(self, tokenizer, model, prompt, device):
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        # get logits
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        logits = model(**inputs).logits
        next_logits = logits[0, -1]
        # get prediction
        pos_id = tokenizer.encode(" positive", add_special_tokens=False)
        assert len(pos_id) == 1, f"Expected single token for ' positive' but got {pos_id}"
        pos_id = pos_id[0]
        neg_id = tokenizer.encode(" negative", add_special_tokens=False)
        assert len(neg_id) == 1, f"Expected single token for ' negative' but got {neg_id}"
        neg_id = neg_id[0]
        pos_logits = next_logits[pos_id]
        neg_logits = next_logits[neg_id]
        pred = 1 if pos_logits > neg_logits else 0
        pred = self.label_map[pred]
        return pred

