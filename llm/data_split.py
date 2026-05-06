import os
import datasets


def split_openwebtext(args):
    source_path = args.data_path + f'/{args.dataset}'
    target_path = args.data_path + f'/{args.dataset}-split'
    if os.path.exists(target_path):
        print('Split dataset already exists:', target_path)
        return
    print('Loading dataset from', source_path)
    dataset = datasets.load_dataset("json", data_dir=source_path, data_files="*.jsonl.zst", chunksize=10**9)['train']
    dataset = dataset.train_test_split(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
    print('Saving splitted dataset to', target_path)
    dataset.save_to_disk(target_path)


def split_bookcorpus(args):
    def _merge_texts(batch, n=50):
        merged = []
        buf = []
        for text in batch["text"]:
            buf.append(text)
            if len(buf) == n:
                merged.append("\n\n".join(buf))
                buf = []
        if len(buf) > 0:
            merged.append("\n\n".join(buf))
        return {"text": merged}

    source_path = args.data_path + f'/{args.dataset}'
    target_path = args.data_path + f'/{args.dataset}-split'
    if os.path.exists(target_path):
        print('Split dataset already exists:', target_path)
        return
    print('Loading dataset from', source_path)
    dataset = datasets.load_from_disk(source_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset['train']
    # merge short texts to longer texts
    print(f'Samples before merge:', len(dataset))
    dataset = dataset.map(_merge_texts, batched=True, batch_size=1000)
    print(f'Samples after merge:', len(dataset))
    # split train and test
    dataset = dataset.train_test_split(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
    print('Saving splitted dataset to', target_path)
    dataset.save_to_disk(target_path)


def split_common(args):
    source_path = args.data_path + f'/{args.dataset}'
    target_path = args.data_path + f'/{args.dataset}-split'
    if os.path.exists(target_path):
        print('Split dataset already exists:', target_path)
        return
    print('Loading dataset from', source_path)
    dataset = datasets.load_from_disk(source_path)
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset[args.split]
    dataset = dataset.train_test_split(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
    print('Saving splitted dataset to', target_path)
    dataset.save_to_disk(target_path)


def merge_aime(args):
    aimes = ['aime24', 'aime25-I', 'aime25-II']
    aime_datasets = []
    for aime in aimes:
        source_path = args.data_path + f'/{aime}'
        dataset = datasets.load_from_disk(source_path)
        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset['test']
        aime_datasets.append(dataset)
    # merge datasets
    dataset = datasets.DatasetDict()
    dataset['train'] = aime_datasets[0]
    dataset['test'] = datasets.concatenate_datasets(aime_datasets[1:])
    # save dataset
    target_path = args.data_path + f'/aime24-25'
    print('Saving merged dataset to', target_path)
    dataset.save_to_disk(target_path)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # job arguments
    parser.add_argument('--data_path', type=str, default='../../huggingface/datasets')
    parser.add_argument('--dataset', type=str, default='bookcorpus_bookcorpus')
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)    
    args = parser.parse_args()

    # merge_aime(args)

    if args.dataset in ['openwebtext2', 'openwebtext']:
        split_openwebtext(args)
    elif args.dataset in ['bookcorpus', 'bookcorpus_bookcorpus']:
        split_bookcorpus(args)
    else:
        split_common(args)
