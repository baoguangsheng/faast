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
from .data import dataset_fns
from .model import load_clip, LinearClassifier
from types import SimpleNamespace


class Trainer:

    def __init__(self, args, model, train_dataset, test_dataset):
        self.args = args
        self.model = model
        self.train_dataset_name = args.dataset
        # split train_dataset
        self.train_dataset, self.valid_dataset = random_split(train_dataset, [0.8, 0.2])
        self.test_dataset = test_dataset
        # count
        self.cur_train_step = 0
        # metrics and log
        self._metrics = defaultdict(list)
        log_dir = self.args.output_path + '/runs/run_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_writer = SummaryWriter(log_dir)

    def train(self, model_file=None):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=self.args.num_epochs // 2, gamma=0.1)
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        progress_bar = tqdm(range(self.args.num_epochs * len(train_loader)), desc='Training')
        print(f'Train with {len(self.train_dataset)} samples, validate with {len(self.valid_dataset)} samples.')
        # training the model
        self.model.to(self.args.device)
        self.model.train()
        best_acc = 0.0
        for epoch in range(self.args.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                loss = self.compute_loss(images, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # count
                self.cur_train_step += 1
                progress_bar.update(1)
                # log
                if self.cur_train_step % 200 == 0:
                    print(f'Step {self.cur_train_step}:', self.log(f'epoch={epoch}'))
            # end of epoch
            best_acc = self.validate_and_save(model_file, best_acc)
            scheduler.step()
        # final validation
        if self.test_dataset is not None:
            print('Final evaluation on test set:')
            self.model.load_from(model_file)
            self.evaluate(self.train_dataset_name.upper() + '-Test', self.test_dataset)

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
        log_obj['acc'] = acc
        # log metrics
        self._update_metrics(log_obj)
        return loss

    def validate_and_save(self, model_file=None, best_acc=0.0):
        acc = self.evaluate(self.train_dataset_name.upper() + '-Valid', self.valid_dataset)
        if acc > best_acc:
            best_acc = acc
            # save model
            if model_file:
                self.model.save_to(model_file)
                print(f'New best model saved with acc={best_acc:.4f}')
                return best_acc                
        print(f'Best acc so far: {best_acc:.4f}')
        return best_acc

    def evaluate(self, prefix, testset):
        self.model.to(self.args.device)
        self.model.eval()
        self._metrics.clear()
        # eval
        test_loader = DataLoader(testset, batch_size=self.args.batch_size,
                                                shuffle=False, num_workers=self.args.num_workers)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                logits = self.model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # log metrics
                log_obj = {}
                self.model.log_metrics(log_obj)
                self._update_metrics(log_obj)
            self._metrics['acc'] = correct / total
            self._metrics['nsamples'] = total
            print(self.log(prefix))
        return correct / total

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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clip, process = load_clip()
    train_dataset, test_dataset = dataset_fns[args.dataset](args.data_path, process)
    model = LinearClassifier(clip, train_dataset.classes)
    model.print_parameters()

    linear_classifier_file = args.output_path + f'/linear_classifier_{args.dataset}.pt'
    if model.has_trainable_parameters():
        print(f'Training classifier...')
        trainer = Trainer(args, model, train_dataset, test_dataset)
        trainer.train(linear_classifier_file)
    else:
        print(f'No trainable parameters, skip training.')
