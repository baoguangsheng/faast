import datetime
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from .data import dataset_fns, FewShotBatchSampler
from .model import load_clip, LinearClassifier
from types import SimpleNamespace


class Trainer:

    def __init__(self, args, model, train_dataset, valid_dataset):
        self.args = args
        self.model = model
        self.train_dataset_name = args.dataset
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = None
        # count
        self.cur_train_step = 0
        # metrics and log
        self._metrics = defaultdict(list)
        log_dir = self.args.output_path + '/runs/run_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_writer = SummaryWriter(log_dir)

    def _get_dataloader(self, name, dataset, num_per_class):
        print(f'Creating {name} batch sampler...')
        labels = dataset.labels
        batch_sampler = FewShotBatchSampler(labels, num_per_class, self.args.num_way, self.args.num_iter)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.args.num_workers)
        return dataloader

    def _split_support_batch(self, images, labels):
        # group indices by label
        label_indices = defaultdict(list)
        for index, label in enumerate(labels):
            label_indices[label.item()].append(index)
        # split 1/8 (at least one) samples per label for validation
        val_indices = []
        train_indices = []
        for label in label_indices:
            indices = label_indices[label]
            random.shuffle(indices)
            n_val = max(1, len(indices) // 8)
            val_indices += indices[:n_val]
            train_indices += indices[n_val:]
        random.shuffle(val_indices)
        random.shuffle(train_indices)
        images0 = images[train_indices]
        labels0 = labels[train_indices]
        images1 = images[val_indices]
        labels1 = labels[val_indices]
        return images0, labels0, images1, labels1

    def _batch_samples(self, images, labels):
        batch_size = self.args.batch_size
        list_images = []
        list_labels = []
        for i in range(0, len(images), batch_size):
            list_images.append(images[i:i+batch_size])
            list_labels.append(labels[i:i+batch_size])
        return list_images, list_labels
    
    def train(self, model_file=None):
        # training the model
        self.model.to(self.args.device)
        self.model.train()
        # few-shot training
        support_loader = self._get_dataloader('support', self.train_dataset, self.args.num_shot)
        query_loader = self._get_dataloader('query', self.valid_dataset, self.args.num_query)
        support_iter = iter(support_loader)
        query_iter = iter(query_loader)
        corrects = []
        totals = []
        accs = []
        for idx in tqdm(range(self.args.num_iter), desc='Evaluating fewshot'):
            self.model.reset_projection()
            # support
            images, labels = next(support_iter)
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            images0, labels0, images1, labels1 = self._split_support_batch(images, labels)
            images0, labels0 = self._batch_samples(images0, labels0)
            # query
            images2, labels2 = next(query_iter)
            images2 = images2.to(self.args.device)
            labels2 = labels2.to(self.args.device)
            images2, labels2 = self._batch_samples(images2, labels2)
            # train for several epochs
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
            scheduler = StepLR(optimizer, step_size=self.args.num_epochs // 2, gamma=0.1)
            best_acc = 0.0
            best_epoch = 0
            test_correct = 0
            test_total = 0
            for epoch in range(self.args.num_epochs):
                self.model.train()
                for images, labels in zip(images0, labels0):
                    loss = self.compute_loss(images, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # count
                self.cur_train_step += 1
                # log
                # print(f'Step {self.cur_train_step}:', self.log(f'epoch={epoch}'))
                scheduler.step()
                # validate
                with torch.no_grad():
                    self.model.eval()
                    logits = self.model(images1)
                    _, predicted = torch.max(logits, 1)
                total1 = labels1.size(0)
                correct1 = (predicted == labels1).sum().item()
                acc1 = correct1 / total1
                # test
                total2 = 0
                correct2 = 0
                for images, labels in zip(images2, labels2):
                    with torch.no_grad():
                        self.model.eval()
                        logits = self.model(images)
                        _, predicted = torch.max(logits, 1)
                    total2 += labels.size(0)
                    correct2 += (predicted == labels).sum().item()
                # save the best
                if acc1 > best_acc:
                    best_acc = acc1
                    best_epoch = epoch
                    test_correct = correct2
                    test_total = total2
                if epoch - best_epoch >= 5:
                    # early stop if no improvement in 5 epochs
                    break
            # log metrics
            log_obj = {'valid_acc': best_acc}
            self._update_metrics(log_obj)
            corrects.append(test_correct)
            totals.append(test_total)
            accs.append(test_correct / test_total)
        # calculate the 95% confidence interval and acc variance
        total = sum(totals)
        acc = float(np.mean(accs))
        acc_std = float(np.std(accs))
        acc_ci95 = 1.96 * acc_std / self.args.num_iter ** 0.5
        self._metrics['nsamples'] = total
        self._metrics['acc'] = round(acc, 4)
        self._metrics['acc_ci95'] = round(acc_ci95, 4)
        self._metrics['acc_std'] = round(acc_std, 4)
        print(self.log(f'{self.args.num_shot}-SHOT-EVAL'))

    def compute_loss(self, images, labels):
        # forward pass
        self.model.train()
        bsz = images.size(0)
        self.model.learn(images, labels)
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        log_obj = {'loss': loss.item()}
        self.model.log_metrics(log_obj)
        # accuracy
        _, predicted = torch.max(logits, 1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        # log metrics
        self._update_metrics(log_obj)
        return loss

    def _update_metrics(self, log_obj):
        for key in log_obj:
            self._metrics[key].append(log_obj[key])

    def log(self, prefix):
        logs = [prefix]
        for key in self._metrics:
            val = np.mean(self._metrics[key])
            self.log_writer.add_scalar(f'{prefix}/{key}', val, global_step=self.cur_train_step)
            logs.append(f'{key}={val:.4f}')
        self._metrics.clear()
        return ', '.join(logs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # job arguments
    parser.add_argument('--output_path', type=str, default='exp_vision/output')
    parser.add_argument('--data_path', type=str, default='exp_vision/data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    # general training arguments
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)    
    parser.add_argument('--device', type=str, default='cuda')
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
    train_dataset, test_dataset = dataset_fns[args.dataset](args.data_path, process)
    model = LinearClassifier(clip, train_dataset.classes)
    model.print_parameters()

    if model.has_trainable_parameters():
        print(f'Training classifier...')
        trainer = Trainer(args, model, train_dataset, test_dataset)
        trainer.train()
    else:
        print(f'No trainable parameters, skip training.')
