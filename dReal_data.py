import torch
import torchvision
from torch.utils.data import *
import torch.distributed as dist
import os
import random
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import numpy as np
from collections.abc import Callable
import torch
from datasets import load_dataset
from torch.utils.data import *
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from tqdm.auto import tqdm


def last_even_num(odd_or_even_num):
    if odd_or_even_num % 2 == 0:
        return odd_or_even_num
    else:
        return odd_or_even_num - 1


class DReal_Dataset_Indices:
    def __init__(self, B, node_cnt, node_idx_map, args=None) -> None:
        super().__init__()
        self.B = B
        self.node_cnt = node_cnt
        self.individual_batch_cnt = node_idx_map.shape[1]
        # self.node_idx_map[i, j] means the original data index for node i in batch j
        # local_node_idx_map = self._get_consistent_node_idx_map(node_idx_map, args)[
        #     args.rank]
        if B == 1:
            self.local_indices = node_idx_map[args.rank].flatten()
        else:
            self.local_indices = node_idx_map[args.rank]

    # def _get_consistent_node_idx_map(self, node_idx_map, args):
    #     # dReal: sync the indices --> make rank 0 the one that distributes partitions
    #     node_idx_map_buffer = node_idx_map
    #     node_idx_map_buffer = node_idx_map_buffer.to(
    #         f'cuda:{args.dev_id}') if args.backend == "nccl" else node_idx_map_buffer
    #     node_idx_map_buffer = node_idx_map_buffer.cpu(
    #     ) if args.backend == "nccl" else node_idx_map_buffer
    #     return node_idx_map_buffer


class DReal_Dataset_Partitioned(DReal_Dataset_Indices):
    def __init__(self, dataset, B, node_cnt, args, device=None) -> None:
        # must shuffle before dividing
        # shuffle, split
        # shuffle before reshape, instead of simply calling, arange => random.shuffle => reshape
        data_idx = np.arange(len(dataset))
        random.shuffle(data_idx)
        total_batch_cnt = last_even_num(len(dataset) // (B * node_cnt)) * node_cnt
        target_len = total_batch_cnt * B
        node_idx_map = torch.tensor(data_idx[:target_len], dtype=torch.int64, device=device).reshape(
            (node_cnt, total_batch_cnt // node_cnt, B))
        super().__init__(B, node_cnt, node_idx_map, args)


def partitioned_dReal_dset_maker(
    dset, B, nodes, args, device=None): return DReal_Dataset_Partitioned(dset, B, nodes, args, device=device)


class DReal_Dataset_RandomSampled(DReal_Dataset_Indices):
    def __init__(self, dataset, B, node_cnt, args, ratio=None) -> None:
        if ratio is None:
            ratio = 1 / node_cnt
        total_datapoint_cnt = last_even_num(len(dataset) // B) * B
        idx_batchify = torch.arange(total_datapoint_cnt, dtype=torch.int).reshape(
            ((len(dataset) // B), B))
        individual_batch_cnt = int((len(dataset) // B) * ratio)
        node_batch_idx_map = torch.tensor([np.random.choice(np.arange(
            idx_batchify.shape[0]), replace=False, size=individual_batch_cnt) for _ in range(node_cnt)])
        node_idx_map = torch.arange(total_datapoint_cnt).reshape(
            (len(dataset) // B, B))[node_batch_idx_map]
        super().__init__(B, node_cnt, node_idx_map, args)


def random_sample_dReal_dset_maker(dset, B, nodes, args, ratio): return DReal_Dataset_RandomSampled(
    dset, B, nodes, args, ratio=ratio)


class DReal_Dataset_Plain(DReal_Dataset_Indices):
    def __init__(self, dataset, B, node_cnt, args) -> None:
        total_datapoint_cnt = last_even_num(len(dataset) // B) * B
        node_idx_map = torch.stack(
            [torch.arange(total_datapoint_cnt).reshape((len(dataset) // B), B)] * node_cnt)
        super().__init__(B, node_cnt, node_idx_map, args)


def plain_dReal_dset_maker(
    dset, B, nodes, args): return DReal_Dataset_Plain(dset, B, nodes, args)


# most helpful if n << class_count
class DReal_Dataset_ClassSplitted(DReal_Dataset_Indices):
    def __init__(self, dataset, B, node_cnt, args, class_count=10) -> None:
        single_node_class = (class_count // node_cnt)
        single_class_node_map = dict()
        single_class_node_class_arr = np.arange(
            0, node_cnt * single_node_class).reshape(node_cnt, single_node_class)
        for node_i in range(node_cnt):
            for cls in single_class_node_class_arr[node_i]:
                single_class_node_map[cls] = node_i

        idx_cls_arr = []
        node_idx_map = [None for _ in range(node_cnt)]
        for i, (_, Y) in enumerate(dataset):
            idx_cls_arr.append((i, Y))

        idx_cls_arr = sorted(idx_cls_arr, key=(
            lambda pair: (pair[1], pair[0])))
        split = len(idx_cls_arr) // node_cnt
        for node_i, start in enumerate(np.arange(0, split * node_cnt, split)):
            node_idx_map[node_i] = [p[0]
                                    for p in idx_cls_arr[start: start + split]]

        node_idx_map = np.array(node_idx_map, dtype=np.int32)
        individual_total_B = (node_idx_map.shape[1] // B)
        individual_total_dp = B * individual_total_B
        node_idx_map = node_idx_map[:, :individual_total_dp].reshape(
            node_cnt, individual_total_B, B)

        super().__init__(B, node_cnt, node_idx_map, args)


def class_splitted_dReal_dset_maker(dset, B, nodes, args, class_count=10): return DReal_Dataset_ClassSplitted(
    dset, B, nodes, args, class_count=class_count)


class DReal_VisionData(Dataset):
    def __init__(self, node_cnt, dset_maker: VisionDataset,
                 dset_addr, train_transform, test_transform, d_dataset_format=partitioned_dReal_dset_maker,
                 download=False, train_B=16, test_B=128, device=None, dtype=torch.float32, args=None, **kw) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.trainset: VisionDataset = dset_maker(
            root=dset_addr, train=True, download=download, transform=train_transform)
        self.testset: VisionDataset = dset_maker(
            root=dset_addr, train=False, download=download, transform=test_transform)

        self.indices: DReal_Dataset_Indices = d_dataset_format(
            self.trainset, train_B, self.node_cnt, args=args, **kw)

        # print(f'{args.rank}: {self.indices.local_indices}')
        self.device = device
        self.dtype = dtype

        self.trainloader = DataLoader(self.trainset, batch_size=test_B)
        self.testloader = DataLoader(self.testset, batch_size=test_B)

        self.images = torch.stack([self.trainset[idx][0] for idx in self.indices.local_indices.view(-1)]).to(
            device=self.device, dtype=self.dtype)
        self.targets = torch.tensor(
            [self.trainset[idx][1] for idx in self.indices.local_indices.view(-1)], device=self.device, dtype=self.dtype)

        self.images = self.images.reshape(
            self.indices.individual_batch_cnt, train_B, *self.images[0].shape)
        self.targets = self.targets.reshape(
            self.indices.individual_batch_cnt, train_B, *self.targets[0].shape)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class DReal_CIFAR10(DReal_VisionData):
    def __init__(self, node_cnt, train_B=16, test_B=64,
                 dset_addr=f'data{os.sep}cifar10-data', d_dataset_format=partitioned_dReal_dset_maker, download=False, device=None, args=None, **kw) -> None:
        cifar10_normalize_transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            cifar10_normalize_transform,
        ]
        )
        test_transform = transforms.Compose([
            transforms.ToTensor(), cifar10_normalize_transform
        ]
        )
        super(DReal_CIFAR10, self).__init__(
            node_cnt, torchvision.datasets.CIFAR10,
            dset_addr, train_transform, test_transform,
            download=download, train_B=train_B, test_B=test_B,
            d_dataset_format=d_dataset_format, device=device, args=args,
            **kw
        )
        self.figure_size_flatten = 3 * 32 * 32
        self.num_classes = 10


class DReal_MNIST(DReal_VisionData):
    def __init__(self, node_cnt, train_B=16, test_B=64,
                 dset_addr=f'data{os.sep}mnist-data', d_dataset_format=partitioned_dReal_dset_maker, download=False, device=None, args=None, **kw) -> None:
        mnist_normalize_transform = transforms.Normalize((0.1307,), (0.3081,))
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            mnist_normalize_transform,
        ]
        )
        test_transform = transforms.Compose([
            transforms.ToTensor(), mnist_normalize_transform
        ]
        )
        super(DReal_MNIST, self).__init__(
            node_cnt, torchvision.datasets.MNIST,
            dset_addr, train_transform, test_transform,
            download=download, d_dataset_format=d_dataset_format,
            train_B=train_B, test_B=test_B,
            device=device, args=args, **kw
        )
        self.figure_size_flatten = 1 * 28 * 28
        self.num_classes = 10


class GLUE:
    glue_task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    def __init__(self, args, exp_config) -> None:
        super().__init__()

        self.raw_datasets = exp_config["raw_datasets"]
        sentence1_key, sentence2_key = self.glue_task_to_keys[args.task_name]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        padding = "max_length"

        label_to_id = exp_config["label_to_id"]

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding,
                               max_length=128, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l]
                                        for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = self.raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        self.trainset = processed_datasets["train"]
        self.testset = processed_datasets["validation_matched" if args.task_name ==
                                          "mnli" else "validation"]
        # DataLoaders creation:
        self.data_collator = default_data_collator


class CReal_GLUE:
    def __init__(self, 
                 args, 
                 train_datadict, 
                 eval_datadict, 
                 node_cnt, 
                 device=None, 
                 d_dataset_format=partitioned_dReal_dset_maker, **kw) -> None:
        self.node_cnt = node_cnt
        self.device = device
        
        length = train_datadict['input_ids'].shape[0]
        self.indices: DReal_Dataset_Indices = d_dataset_format(
            torch.arange(length, device=device), 
            1, self.node_cnt, args=args, device=device, **kw
        )

        self.train_datadict = train_datadict
        self.eval_datadict = eval_datadict

        self.node_cnt = node_cnt
        self.device = device

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return {
            k: v[self.indices.local_indices[index]]
                for k, v in self.train_datadict.items()
        }

    def eval_item(self, index, is_train=True):
        datadict = self.train_datadict if is_train else self.eval_datadict
        ret_dict = {
            k: v[index] for k, v in datadict.items()
        }
        ret_dict['output_logits'] = True
        return ret_dict
    
    def eval_length(self, is_train=True):
        datadict = self.train_datadict if is_train else self.eval_datadict
        return datadict['input_ids'].shape[0]


class CReal_GLUE_Embeddings:
    @torch.no_grad()
    def __init__(self, args, exp_config, node_cnt, model, device=None, d_dataset_format=partitioned_dReal_dset_maker, **kw) -> None:
        self.node_cnt = node_cnt
        self.device = device
        # self.glue = GLUE(args, exp_config)
        train_embeddings_addr = f'data{os.sep}glue-data{os.sep}{args.task_name}-bertbase-train-embedding.pt'
        test_embeddings_addr = f'data{os.sep}glue-data{os.sep}{args.task_name}-bertbase-test-embedding.pt'

        if not (os.path.exists(train_embeddings_addr) and os.path.exists(test_embeddings_addr)):
            model = model()
            model.eval()
            trainset_raw = list(DataLoader(
                self.glue.trainset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=128))
            testset_raw = list(DataLoader(
                self.glue.testset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=32))
            self.trainset_embeddings = []
            for batch in tqdm(trainset_raw):
                embeddings = model(
                    input_ids=batch['input_ids'].to(device=device),
                    token_type_ids=batch['token_type_ids'].to(device=device),
                    attention_mask=batch['attention_mask'].to(device=device),
                )[1]
                self.trainset_embeddings.append(embeddings)
            self.trainset_embeddings = torch.vstack(self.trainset_embeddings)
            torch.save(self.trainset_embeddings, train_embeddings_addr)
            self.testset_embeddings = []
            for batch in tqdm(testset_raw):
                embeddings = model(
                    input_ids=batch['input_ids'].to(device=device),
                    token_type_ids=batch['token_type_ids'].to(device=device),
                    attention_mask=batch['attention_mask'].to(device=device),
                )[1]
                self.testset_embeddings.append(embeddings)
            self.testset_embeddings = torch.vstack(self.testset_embeddings)
            torch.save(self.testset_embeddings, test_embeddings_addr)
        else:
            self.trainset_embeddings = torch.load(
                train_embeddings_addr, map_location=torch.device('cpu'))
            self.testset_embeddings = torch.load(
                test_embeddings_addr, map_location=torch.device('cpu'))

        self.trainset_labels = torch.tensor(load_dataset('glue', args.task_name)['train']['label'], 
                                            dtype=torch.int64, device=self.device)
        self.testset_labels = torch.tensor(load_dataset('glue', args.task_name)[
                                           'validation']['label'], dtype=torch.int64, device=self.device)
        self.trainset = [(self.trainset_embeddings[i], self.trainset_labels[i])
                         for i in range(len(self.trainset_embeddings))]
        self.testset = [(self.testset_embeddings[i], self.testset_labels[i])
                        for i in range(len(self.testset_embeddings))]
        self.indices: DReal_Dataset_Indices = d_dataset_format(
                self.trainset_embeddings, 1, self.node_cnt, args=args, **kw)
        self.trainloader = DataLoader(self.trainset, batch_size=args.test_B)
        self.testloader = DataLoader(self.testset, batch_size=args.test_B)
        
        self.node_cnt = node_cnt
        self.device = device

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        embeddings = self.trainset_embeddings[self.indices.local_indices[index]].to(
            device=self.device)
        labels = self.trainset_labels[self.indices.local_indices[index]].to(
            device=self.device)
        return embeddings, labels


class Dataset_M4(Dataset):
    def __init__(self,
                 input_length,  # num of input steps
                 output_length,  # forecasting horizon
                 freq,  # The frequency of time series
                 train_data_addr="data/M4/train.npy",  # path to numpy data files
                 # for testing mode, we need to load both train and test data
                 test_data_addr="data/M4/test.npy",
                 mode="train",  # train, validation or test
                 expand_dim=False,  # whether expand last dimension
                 seed=0,
                 device=None
                 ):
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.expand_dim = expand_dim
        self.device = device
        # Load training set
        self.train_data = np.load(train_data_addr, allow_pickle=True)
        self.data_lsts = self.train_data.item().get(freq)

        # First do global standardization
        self.ts_means, self.ts_stds = [], []
        for i in range(len(self.data_lsts)):
            avg, std = np.mean(self.data_lsts[i]), np.std(self.data_lsts[i])
            self.ts_means.append(avg)
            self.ts_stds.append(std)
            self.data_lsts[i] = (self.data_lsts[i] - avg) / std

        if mode == "test":
            self.test_lsts = np.load(
                test_data_addr, allow_pickle=True).item().get(freq)
            for i in range(len(self.test_lsts)):
                self.test_lsts[i] = (self.test_lsts[i] -
                                     self.ts_means[i])/self.ts_stds[i]
            self.ts_indices = [i for i in range(len(self.test_lsts))]

        elif mode == "train" or "valid":
            # shuffle slices before split
            self.ts_indices = [(i, j) for i in range(len(self.data_lsts))
                               for j in range(0, len(self.data_lsts[i]) - input_length - output_length, 3)]
            np.random.RandomState(0).shuffle(self.ts_indices)

            # 80%-20% train-validation split
            if mode == "train":
                self.ts_indices = self.ts_indices[:int(
                    len(self.ts_indices)*0.9)]
            elif mode == "valid":
                self.ts_indices = self.ts_indices[int(
                    len(self.ts_indices)*0.9):]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            x = self.data_lsts[index][-self.input_length:]
            y = self.test_lsts[index]
        else:
            i, j = self.ts_indices[index]
            x = self.data_lsts[i][j:j+self.input_length]
            y = self.data_lsts[i][j+self.input_length: j +
                                  self.input_length+self.output_length]

        if self.expand_dim:
            return torch.from_numpy(x).float().unsqueeze(-1).to(self.device),  torch.from_numpy(y).float().unsqueeze(-1).to(self.device)
        return torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).float().to(self.device)


class C_M4_Dataset(Dataset):
    def __init__(self, args, node_cnt, input_length, output_length, freq, train_B, device=None, d_dataset_format=partitioned_dReal_dset_maker) -> None:
        self.train_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                        freq=freq, mode="train", expand_dim=False, device=device)
        self.val_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                      freq=freq, mode="valid", expand_dim=False, device=device)
        self.test_dataset = Dataset_M4(
            input_length=input_length, output_length=13, freq=freq, mode="test", expand_dim=False, device=device)

        self.indices = d_dataset_format(
            self.train_dataset, train_B, node_cnt, args)
        self.train_loader_eval = DataLoader(
            self.train_dataset, batch_size=1024, shuffle=False)
        self.valid_loader = DataLoader(
            self.val_dataset, batch_size=1024, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, shuffle=False)
        self.device = device

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, idx):
        mapped_idx = self.indices.local_indices[idx]
        inps, tgts = self.train_dataset[mapped_idx]
        if len(inps.shape) > 1:
            return inps, tgts
        else:
            return inps.unsqueeze(0).unsqueeze(-1), tgts.unsqueeze(0).unsqueeze(-1)

class dist_CIFAR100(Dataset):
    def __init__(self, config, args, node_cnt, microbatch, test_B=128, device=torch.device('cuda')) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.device = device
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_path = config.DATA.DATA_PATH
        self.transform_train = build_transform(True, config)
        self.transform_test = build_transform(False, config)
        
        if not os.path.exists(self.data_path):
            dset = []
            for _ in range(5):
                dset.extend(list(DataLoader(
                    torchvision.datasets.CIFAR100(root='./data', train=True, transform=self.transform_train, 
                                                download=True), batch_size=128, shuffle=True)))
            images = torch.vstack([p[0] for p in dset])
            targets = torch.cat([p[1] for p in dset])
            torch.save((images, targets), self.data_path)
            self.images = images
            self.targets = targets
        else:
            self.images, self.targets = torch.load(self.data_path, map_location=device)

        self.indices = partitioned_dReal_dset_maker(
            self.images, microbatch, self.node_cnt, args=args)
        self.indices.local_indices = self.indices.local_indices.view(-1)
        self.indices.individual_batch_cnt = len(self.indices.local_indices)
        self.trainloader = DataLoader(
            torchvision.datasets.CIFAR100(root='./data', train=True, transform=self.transform_train,\
                download=True), batch_size=128, shuffle=True
        )
        self.testloader = DataLoader(
            torchvision.datasets.CIFAR100(
            root='./data', train=False, transform=self.transform_test, batch_size=test_B, shuffle=False)
            )

        self.images = self.images[self.indices.local_indices].to(
            device=self.device)
        self.targets = self.targets[self.indices.local_indices].to(
            device=self.device, dtype=torch.int64)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.images[index], self.targets[index]
        


class CIFAR10_ResNet_Data(Dataset):
    def __init__(self, args, node_cnt, microbatch, test_B=128, device=torch.device('cuda')) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.device = device
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trainset_path = 'cifar10-5fold-img-target-pair.pt'
        if not os.path.exists(trainset_path):
            dset = []
            for _ in range(5):
                dset.extend(list(DataLoader(
                    torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ]), download=True), batch_size=128, shuffle=True)))
            images = torch.vstack([p[0] for p in dset])
            targets = torch.cat([p[1] for p in dset])
            torch.save((images, targets), trainset_path)
            self.images = images
            self.targets = targets
        else:
            self.images, self.targets = torch.load(trainset_path, map_location=device)

        self.indices = partitioned_dReal_dset_maker(
            self.images, microbatch, self.node_cnt, args=args)
        self.indices.local_indices = self.indices.local_indices.view(-1)
        self.indices.individual_batch_cnt = len(self.indices.local_indices)
        self.trainloader = DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True), batch_size=128, shuffle=True
        )
        self.testloader = DataLoader(
            torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])), batch_size=test_B, shuffle=False)

        self.images = self.images[self.indices.local_indices].to(
            device=self.device)
        self.targets = self.targets[self.indices.local_indices].to(
            device=self.device, dtype=torch.int64)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.images[index], self.targets[index]
    
    
    
def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
