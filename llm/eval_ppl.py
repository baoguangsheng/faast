import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, random_split
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, default_data_collator
from collections import defaultdict
import datetime
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace


class Evaluator:

    def __init__(self, args, model, tokenizer, support_dataset, query_dataset):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # learn dataset
        self.dataset_name = args.dataset
        self.support_dataset = support_dataset
        self.query_dataset = query_dataset
        # count
        self.cur_learn_step = 0
        # metrics and log
        self._metrics = defaultdict(list)
        log_dir = self.args.output_path + '/runs/run_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_writer = SummaryWriter(log_dir)

    def _update_metrics(self, log_obj):
        for key in log_obj:
            self._metrics[key].append(log_obj[key])

    def log(self, prefix):
        logs = [prefix]
        for key in self._metrics:
            val = np.mean(self._metrics[key])
            self.log_writer.add_scalar(f'{prefix}/{key}', val, global_step=self.cur_learn_step)
            logs.append(f'{key}={val:.4f}')
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
        return 1024

    def eval_ppl(self, memory_model=False):
        # Sliding Window Parameters
        max_length = self.get_max_length()
        stride = max_length  # A smaller stride gives more context and better results

        encodings = self.tokenizer("\n\n".join(self.query_dataset["text"]), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        # Evaluation Loop
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # tokens we actually want to predict
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
            target_ids = input_ids.clone()
            
            # We only want to compute loss for the new tokens in this window (trg_len)
            # Mask out previous context tokens by setting them to -100
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # loss is the average nll per token in the target_ids
                neg_log_likelihood = outputs.loss * trg_len
                # log
                if memory_model:
                    log_obj = {}
                    self.model.log_metrics(log_obj)
                    self._update_metrics(log_obj)

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        # print log
        print(self.log('Eval-Memory') if memory_model else self.log('Eval-Basic'))

        # Final Perplexity Calculation
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        print(f"Perplexity of {self.args.model_path} on WikiText-103: {ppl.item():.2f}")

    def learn_memory(self):
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = tokenizer.eos_token
        self.model.reset_projection()
        # Sliding Window Parameters
        max_length = self.get_max_length()
        stride = max_length  # A smaller stride gives more context and better results

        print(f'Learning memory from {len(self.support_dataset)} samples...')
        # do NOT shuffle here, which breaks the text continuity in the batch
        data_loader = DataLoader(self.support_dataset, batch_size=64)
        progress = tqdm(data_loader, desc='Learning batches')
        learn_time = 0.0
        for idx, batch in enumerate(progress):
            encodings = self.tokenizer("\n\n".join(batch["text"]), return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
            # packing into 1024
            batch_input_ids = []
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.args.device)
                if input_ids.size(1) < stride:
                    # pad input_ids with -100 using F.pad
                    pad_len = stride - input_ids.size(1)
                    input_ids = F.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                assert input_ids.size(1) == stride
                batch_input_ids.append(input_ids)
            # learning
            input_ids = torch.cat(batch_input_ids, dim=0)
            attn_mask = input_ids.ne(tokenizer.pad_token_id).long()
            with torch.no_grad():
                start_time = datetime.datetime.now()
                inputs = {'input_ids': input_ids, 'attention_mask': attn_mask}
                self.model.learn(**inputs)
                end_time = datetime.datetime.now()
                learn_time += (end_time - start_time).total_seconds()
            progress.set_postfix(mem_size=self.model.memory_size/1024, learn_time=learn_time/60)
            # log
            log_obj = {}
            self.model.log_metrics(log_obj)
            self._update_metrics(log_obj)
            # eval
            if idx % 200 == 0:
                self.eval_ppl(memory_model=True)
        # print log
        print(self.log('Learn-Memory'))



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='exp_llm/output')
    parser.add_argument('--model_path', type=str, default='../../huggingface/models/gpt2')
    parser.add_argument('--data_path', type=str, default='../../huggingface/datasets/Salesforce_wikitext.wikitext-103-raw-v1')
    parser.add_argument('--dataset', type=str, default='wikitext-103')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--learn_samples', type=int, default=None)
    parser.add_argument('--update_alpha', type=float, default=0.8)
    parser.add_argument('--update_size', type=int, default=1024*8)
    parser.add_argument('--memory_size', type=int, default=1024*64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # prepare dataset
    dataset = datasets.load_from_disk(args.data_path)
    trainset = dataset['train']
    testset = dataset['test']
    if args.learn_samples:
        trainset = trainset.select(range(args.learn_samples))

    # prepare model
    assert args.memory_size >= args.update_size, "memory_size should be larger than update_size"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.config.use_cache = False
    model.config.reader_update_alpha = args.update_alpha
    model.config.reader_update_size = args.update_size
    model.set_max_memory_size(args.memory_size)
    model = model.to(getattr(torch, args.dtype))
    model.to(args.device)
    model.eval()

    print(f'Model loaded from {args.model_path} to {args.device}')
    evaluator = Evaluator(args, model, tokenizer, trainset, testset)
    # evaluator.print_parameters()

    if getattr(model, 'require_learning', False):
        evaluator.learn_memory()
        evaluator.eval_ppl(memory_model=True)
    else:
        evaluator.eval_ppl(memory_model=False)
