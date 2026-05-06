import os
import random
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, random_split
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, default_data_collator
from collections import defaultdict
import datetime
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from .few_shot.sentiment_analysis_task import SentimentAnalysisTask
from .few_shot.machine_translation_task import MachineTranslationTask
from .few_shot.math_reasoning_task import MathReasoningTask
from .few_shot.long_icl_bench_task import LongICLBenchTask
from .few_shot.task_base import TaskBase



DATASET_TASKS = {
    'sst2': SentimentAnalysisTask,
    'imdb': SentimentAnalysisTask,
    'iwslt2017-en-de': MachineTranslationTask,
    'iwslt2017-de-en': MachineTranslationTask,
    'iwslt2017-en-fr': MachineTranslationTask,
    'iwslt2017-fr-en': MachineTranslationTask,
    'gsm1k-split': MathReasoningTask,
    'LongICLBench.GoEmotion': LongICLBenchTask,
    'LongICLBench.BANKING77': LongICLBenchTask,
    'LongICLBench.FewNERD': LongICLBenchTask,
    'LongICLBench.TacRED': LongICLBenchTask,
    'LongICLBench.Discovery': LongICLBenchTask,
    'LongICLBench.DialogRE': LongICLBenchTask,
}

# Special settings for support and query splits other than train and test
DATASET_SPLITS = {
    'sst2': {'support': 'train', 'query': 'validation'},
}

# Alpha is determined by the task complexity, model capability, noise level, and data size.
# Here the n is the number of shots. Generally, a proper alpha should increase with more data.
MODEL_DATASET_ALPHAS = {
    'gpt2-xl': {
        'sst2': lambda n: 0.9 + 0.1 * min(n, 100) / 100,
        'imdb': lambda n: 0.6 + 0.1 * min(n, 100) / 100,
    },
    'Qwen2.5-3B-Instruct': {
        'iwslt2017-en-de': lambda n: 0.5 + 0.3 * min(n, 100) / 100,
        'iwslt2017-de-en': lambda n: 0.6 + 0.3 * min(n, 100) / 100,
        'iwslt2017-en-fr': lambda n: 0.5 + 0.3 * min(n, 100) / 100,
        'iwslt2017-fr-en': lambda n: 0.6 + 0.3 * min(n, 100) / 100,
    },
    'Qwen2.5-7B-Instruct': {
        'iwslt2017-en-de': lambda n: 1.0 + 0.2 * min(n, 100) / 100,
        'iwslt2017-de-en': lambda n: 0.8 + 0.2 * min(n, 100) / 100,
        'iwslt2017-en-fr': lambda n: 0.6 + 0.2 * min(n, 100) / 100,
        'iwslt2017-fr-en': lambda n: 0.9 + 0.2 * min(n, 100) / 100,
        'LongICLBench.GoEmotion': lambda n: 0.8,
        'LongICLBench.BANKING77': lambda n: 1.0,
        'LongICLBench.FewNERD': lambda n: 0.8,
        'LongICLBench.TacRED': lambda n: 0.9,
        'LongICLBench.Discovery': lambda n: 0.6,
        'LongICLBench.DialogRE': lambda n: 0.7,
    },
    'Llama-3-8B-Instruct': {
        'iwslt2017-en-de': lambda n: 1.0 + 0.2 * min(n, 100) / 100,
        'iwslt2017-de-en': lambda n: 0.8 + 0.2 * min(n, 100) / 100,
        'iwslt2017-en-fr': lambda n: 0.6 + 0.2 * min(n, 100) / 100,
        'iwslt2017-fr-en': lambda n: 0.9 + 0.2 * min(n, 100) / 100,
        'LongICLBench.GoEmotion': lambda n: 0.8,
        'LongICLBench.BANKING77': lambda n: 1.0,
        'LongICLBench.FewNERD': lambda n: 0.8,
        'LongICLBench.TacRED': lambda n: 0.9,
        'LongICLBench.Discovery': lambda n: 0.6,
        'LongICLBench.DialogRE': lambda n: 0.7,
    }
}



class Evaluator:

    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.tokenizer, self.model = self._load_model(args)
        chat = args.model_path.lower().find('instruct') > 0
        self.task = DATASET_TASKS[args.dataset](args.dataset, chat)
        if getattr(self.task, 'load_datasets', None):
            self.support_dataset, self.query_dataset = self.task.load_datasets(args)
        else:
            self.support_dataset, self.query_dataset = self._load_datasets(args)
        # count
        self.cur_learn_step = 0
        # metrics and log
        self._metrics = defaultdict(list)

    def _load_datasets(self, args):
        # prepare dataset
        dataset = datasets.load_from_disk(args.data_path + f'/{args.dataset}')
        support_split = 'train'
        query_split = 'test'
        if args.dataset in DATASET_SPLITS:
            support_split = DATASET_SPLITS[args.dataset]['support']
            query_split = DATASET_SPLITS[args.dataset]['query']
        supportset = dataset[support_split]
        queryset = dataset[query_split]
        if args.nsamples:
            # random select supportset
            supportset = supportset.shuffle(seed=args.seed)
            supportset = supportset.select(range(min(len(supportset), args.nsamples)))
            # random select queryset; if it is not enough, repeat it.
            need_samples = args.nsamples
            need_sets = []
            queryset = queryset.shuffle(seed=args.seed)
            while need_samples >= len(queryset):
                need_sets.append(queryset)
                need_samples -= len(queryset)
            if need_samples > 0:
                need_sets.append(queryset.select(range(need_samples)))
            queryset = datasets.concatenate_datasets(need_sets)
        return supportset, queryset

    def _get_nshot_name(self, args):
        if args.job == 'full':
            return 'full'
        return str(args.num_shot)
    
    def _get_nshot_scale(self, args):
        if args.job == 'full':
            return 100
        return args.num_shot if args.num_shot is not None else 10
    
    def _get_basemodel_name(self, args):
        model_name = [part for part in args.model_path.split('/') if part.strip()][-1]
        for key in MODEL_DATASET_ALPHAS:
            if key in model_name:
                return key
        raise ValueError(f'Cannot find basemodel name from {model_name}.')
        
    def _load_model(self, args):
        # prepare model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token 
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        model.config.use_cache = False
        # customize rtol for different datasets
        require_learning = getattr(model, 'require_learning', False)
        nshot_name = self._get_nshot_name(args)
        basemodel_name = self._get_basemodel_name(args)
        if require_learning:
            model.config.reader_update_size = args.update_size
            model.set_max_memory_size(args.memory_size)
            # set update alpha based on dataset and nshot
            default_alpha = model.config.reader_update_alpha
            alpha = MODEL_DATASET_ALPHAS[basemodel_name][args.dataset](self._get_nshot_scale(args))
            model.config.reader_update_alpha = alpha
            print(f'Change alpha from {default_alpha} to {model.config.reader_update_alpha} for dataset {args.dataset} {nshot_name}-shot')
        # set dtype and device
        model = model.to(getattr(torch, args.dtype))
        model.to(args.device)
        model.eval()
        return tokenizer, model

    def _update_metrics(self, log_obj):
        for key in log_obj:
            self._metrics[key].append(log_obj[key])

    def log(self, prefix):
        logs = [prefix]
        for key in self._metrics:
            val = self._metrics[key]
            if isinstance(val, list):
                val = np.mean(val)
                logs.append(f'{key}={val:.4f}')
            else:
                logs.append(f'{key}={val}')
        self._metrics.clear()
        return ', '.join(logs)
    
    def print_parameters(self):
        print('Parameters:')
        nparams = 0
        for name, param in self.model.named_parameters():
            print(name, f'({param.norm()})')
            nparams += param.numel()
        print(f'Total: {nparams/1024/1024:.2f}M')

    def get_max_length(self):
        if hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'max_length'):
            return self.model.config.max_length
        else:
            raise ValueError('Cannot determine max length from model config.')

    def evaluate_fewshot(self):
        self.model.to(self.args.device)
        self.model.eval()
        self._metrics.clear()
        # learn
        require_learning = getattr(self.model, 'require_learning', False)
        prefix = f'{self.dataset_name.upper()}-EVAL-{self.args.num_shot}shot'
        support_dataset = self.support_dataset
        query_dataset = self.query_dataset
        with torch.no_grad():
            self.cur_learn_step = 0
            self.task.reset_metric()
            progress = tqdm(range(len(query_dataset)), desc=f'Evaluating {self.args.num_shot}-shot require_learning={require_learning}')
            for idx in progress:
                query_item = query_dataset[idx]
                support_items = self.task.sample_few_shot(query_item, support_dataset, self.args.num_shot)
                if require_learning:
                    # build prompts
                    support_prompts = [self.task.build_prompt(self.tokenizer, query_item=item, include_response=True) for item in support_items]
                    query_prompt = self.task.build_prompt(self.tokenizer, query_item=query_item)
                    # learn support prompts
                    print(f'LEARN:\n{support_prompts[0]}')
                    inputs = self.tokenizer(support_prompts, return_tensors="pt", padding=True, truncation=True).to(self.args.device)
                    self.model.reset_projection()
                    outputs = self.model.learn(**inputs)
                    # log metrics
                    log_obj = {}
                    self.model.log_metrics(log_obj)
                    # print('Learning:', log_obj)
                    progress.set_postfix(mem_size=self.model.memory_size/1024)
                else:
                    # build prompts
                    query_prompt = self.task.build_prompt(self.tokenizer, query_item=query_item, support_items=support_items)
                # complete query prompt
                pred = self.task.predict(self.tokenizer, self.model, query_prompt, self.args.device)
                self.task.update_metric(query_item, pred)
                print('Metric:', self.task.get_metric())
                # log metrics
                if require_learning:
                    log_obj = {}
                    self.model.log_metrics(log_obj)
                    self._update_metrics(log_obj)
                # count step
                self.cur_learn_step += 1
            # calculate the 95% confidence interval
            self._metrics.update(self.task.get_metric())
            print(self.log(prefix))

    def evaluate_full(self, memory_file=None, skip_learning=False):
        if getattr(self.model, 'require_learning', False) and not skip_learning:
            self._learn(self.dataset_name.upper() + '-LEARN', self.support_dataset)
        self._evaluate(self.dataset_name.upper() + '-EVAL-full', self.query_dataset)
        # save memory
        if memory_file is not None:
            self.model.save_memory(memory_file)
            print(f'Saved memory to', memory_file)

    def _learn(self, prefix, dataset):
        self.model.to(self.args.device)
        self.model.eval()
        self.model.reset_projection()
        self._metrics.clear()
        # learn
        data_loader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True, collate_fn=lambda x: x)
        with torch.no_grad():
            data_iter = iter(data_loader)
            self.cur_learn_step = 0
            progress = tqdm(data_iter, desc='Learning')
            for batch in progress:
                prompts = [self.task.build_prompt(self.tokenizer, query_item=item, include_response=True) for item in batch]
                print(f'LEARN:\n{prompts[0]}')                
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.args.device)
                outputs = self.model.learn(**inputs)
                progress.set_postfix(mem_size=self.model.memory_size/1024)
                # log metrics
                log_obj = {}
                self.model.log_metrics(log_obj)
                self._update_metrics(log_obj)
                print(self.log(prefix))
                # count step
                self.cur_learn_step += 1

    def _evaluate(self, prefix, dataset):
        self.model.to(self.args.device)
        self.model.eval()
        self._metrics.clear()
        # eval
        require_learning = getattr(self.model, 'require_learning', False)
        with torch.no_grad():
            self.task.reset_metric()
            num_iter = len(dataset)
            for idx in tqdm(range(num_iter), desc='Evaluating'):
                query_item = dataset[idx]
                query_prompt = self.task.build_prompt(self.tokenizer, query_item=query_item)
                pred = self.task.predict(self.tokenizer, self.model, query_prompt, self.args.device)
                self.task.update_metric(query_item, pred)
                print('Metric:', self.task.get_metric())
                # log metrics
                if require_learning:
                    log_obj = {}
                    self.model.log_metrics(log_obj)
                    self._update_metrics(log_obj)
            # calculate the 95% confidence interval
            self._metrics.update(self.task.get_metric())
            print(self.log(prefix))



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--job', type=str, default='fewshot', choices=['fewshot', 'full'])       
    parser.add_argument('--output_path', type=str, default='exp_llm/output')
    parser.add_argument('--model_path', type=str, default='../../huggingface/models/gpt2-xl')
    parser.add_argument('--data_path', type=str, default='../../huggingface/datasets')
    parser.add_argument('--dataset', type=str, default='gsm8k')  # sst2, imdb, iwslt2017-en-de, gsm8k
    parser.add_argument('--num_shot', type=int, default=None)
    parser.add_argument('--nsamples', type=int, default=None)
    parser.add_argument('--update_size', type=int, default=1024*8)
    parser.add_argument('--memory_size', type=int, default=1024*64)    
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)      
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Model loaded from {args.model_path} to {args.device}')
    evaluator = Evaluator(args)
    # evaluator.print_parameters()

    if args.job == 'fewshot':
        print(f'Evaluating fewshot on {args.dataset} dataset ...')
        evaluator.evaluate_fewshot()

    elif args.job == 'full':
        print(f'Evaluating full on {args.dataset} dataset ...')
        evaluator.evaluate_full()

    else:
        raise ValueError(f'Unknown job type: {args.job}')
    
