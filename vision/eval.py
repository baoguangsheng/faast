import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import datetime
from torch.utils.tensorboard import SummaryWriter
from .data import dataset_fns, FewShotBatchSampler
from .model import load_clip, ClipClassifier, LinearClassifier, MemoryClassifier
from types import SimpleNamespace


DATASET_N0FNS = {
    'cifar10': lambda N: 400, # better choice: np.clip(4*N, 200, 1200),
    'mini-imagenet': lambda N: 800, # better choice: np.clip(10*N, 400, 6000),
    'mini-imagenet-id': lambda N: 0,  # no prior for non-human-readable labels
}


class Evaluator:

    def __init__(self, args, model, support_dataset, query_dataset):
        self.args = args
        self.model = model
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

    def evaluate_full(self, memory_file=None, skip_learning=False):
        if self.model.require_learning and not skip_learning:
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
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        with torch.no_grad():
            data_iter = iter(data_loader)
            self.cur_learn_step = 0
            for images, labels in tqdm(data_iter, desc='Learning'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                self.model.learn(images, labels)
                # log metrics
                log_obj = {}
                self.model.log_metrics(log_obj)
                self._update_metrics(log_obj)
                # count step
                self.cur_learn_step += 1
            print(self.log(prefix))

    def _evaluate(self, prefix, dataset):
        self.model.to(self.args.device)
        self.model.eval()
        self._metrics.clear()
        # eval
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        with torch.no_grad():
            correct = 0
            total = 0
            data_iter = iter(data_loader)
            for images, labels in tqdm(data_iter, desc='Evaluating'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                logits = self.model.forward(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # log metrics
                log_obj = {}
                self.model.log_metrics(log_obj)
                self._update_metrics(log_obj)
            self._metrics['nsamples'] = total
            # calculate the 95% confidence interval
            acc = correct / total
            ci95 = 1.96 * np.sqrt(acc * (1 - acc) / total)
            self._metrics['acc'] = acc
            self._metrics['ci95'] = ci95
            print(self.log(prefix))
  
    def _get_dataloader(self, name, dataset, num_per_class):
        print(f'Creating {name} batch sampler...')
        labels = dataset.labels
        batch_sampler = FewShotBatchSampler(labels, num_per_class, self.args.num_way, self.args.num_iter)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.args.num_workers)
        return dataloader

    def evaluate_fewshot(self):
        self.model.to(self.args.device)
        self.model.eval()
        self._metrics.clear()
        # learn
        prefix = f'EVAL-{self.args.num_shot}shot'
        support_dataset = self.support_dataset
        query_dataset = self.query_dataset
        support_loader = self._get_dataloader('support', support_dataset, self.args.num_shot)
        query_loader = self._get_dataloader('query', query_dataset, self.args.num_query)
        with torch.no_grad():
            accs = []
            corrects = []
            totals = []
            self.cur_learn_step = 0
            support_iter = iter(support_loader)
            query_iter = iter(query_loader)
            for idx in tqdm(range(self.args.num_iter), desc='Evaluating fewshot'):
                if self.model.require_learning:
                    self.model.reset_projection()
                    # support
                    images, labels = next(support_iter)
                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)
                    self.model.learn(images, labels)
                # query
                images, labels = next(query_iter)
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                logits = self.model.forward(images)
                _, predicted = torch.max(logits, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                totals.append(total)
                corrects.append(correct)
                accs.append(correct / total)
                # log metrics
                log_obj = {}
                self.model.log_metrics(log_obj)
                self._update_metrics(log_obj)
                # count step
                self.cur_learn_step += 1
            # calculate the 95% confidence interval and acc variance
            total = sum(totals)
            correct = sum(corrects)
            acc = float(np.mean(accs))
            acc_std = float(np.std(accs))
            acc_ci95 = 1.96 * acc_std / self.args.num_iter ** 0.5
            self._metrics['nsamples'] = total
            self._metrics['acc'] = round(acc, 4)
            self._metrics['acc_ci95'] = round(acc_ci95, 4)
            self._metrics['acc_std'] = round(acc_std, 4)
            print(self.log(prefix))


def load_classifier(args, clip, classes):
    print(f'Loading {args.classifier} classifier with {len(classes)} classes...')

    if args.classifier == 'clip':
        model = ClipClassifier(clip, classes)
    elif args.classifier == 'linear':
        linear_classifier_file = args.output_path + f'/linear_classifier_{args.dataset}.pt'
        model = LinearClassifier(clip, classes)
        model.load_from(linear_classifier_file)
    elif args.classifier == 'memory':
        config = SimpleNamespace(**{'reader_type': args.reader_type})
        model = MemoryClassifier(config, clip, classes)
        model.n0_fn = DATASET_N0FNS[args.dataset]
    else:
        raise ValueError(f'Unknown classifier type: {args.classifier}')
    return model


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # job arguments
    parser.add_argument('--job', type=str, default='fewshot', choices=['fewshot', 'full'])   
    parser.add_argument('--output_path', type=str, default='exp_vision/output')
    parser.add_argument('--model_path', type=str, default='exp_vision/models')
    parser.add_argument('--data_path', type=str, default='exp_vision/data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)    
    # classifier arguments
    parser.add_argument('--classifier', type=str, default='memory', choices=['clip', 'linear', 'memory'])
    parser.add_argument('--reader_type', type=str, default='inverse', choices=['softmax', 'knn', 'linear', 'inverse'])
    # few-shot arguments
    parser.add_argument('--num_iter', type=int, default=600)
    parser.add_argument('--num_way', type=int, default=None)  # default to use all classes
    parser.add_argument('--num_shot', type=int, default=1)
    parser.add_argument('--num_query', type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clip, process = load_clip()
    support_dataset, query_dataset = dataset_fns[args.dataset](args.data_path, process)
    model = load_classifier(args, clip, support_dataset.classes)
    evaluator = Evaluator(args, model, support_dataset, query_dataset)

    if args.job == 'fewshot':
        print(f'Evaluating {args.num_shot}-shot on {args.classifier} classifier ...')
        if args.num_shot > 0:
            evaluator.evaluate_fewshot()
        else:
            evaluator.evaluate_full(skip_learning=True)

    elif args.job == 'full':
        print(f'Evaluating full on {args.classifier} classifier ...')
        evaluator.evaluate_full()

    else:
        raise ValueError(f'Unknown job type: {args.job}')
    
