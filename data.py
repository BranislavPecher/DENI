import torch
import copy
import os
import math
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import RandomSampler, DataLoader, Dataset

class SeededRandomSampler(RandomSampler):

    def __init__(self, dataset, replacement=False, num_samples=None, seed=0):
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
        super(SeededRandomSampler, self).__init__(dataset, replacement, num_samples)
        self.dataset = dataset

    def __iter__(self):
        size = len(self.dataset)

        old_state = torch.get_rng_state()
        torch.set_rng_state(self.state)

        if self.replacement:
            iterator = iter(torch.randint(high=size, size=(self.num_samples,), dtype=torch.int64).tolist())
        else:
            iterator = iter(torch.randperm(size).tolist())

        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
        return iterator


class DatasetLoader():

    def __init__(self, name, batch_size, dataset, shuffle_train_seed=0):
        self.name = name
        self.batch_size = batch_size
        self.shuffle_train_seed = shuffle_train_seed
        self.train_dataset = copy.deepcopy(dataset)
        self.train_dataset.train = True

        self.test_dataset = copy.deepcopy(dataset)
        self.test_dataset.train = False

    def trainloader(self):
        sampler = SeededRandomSampler(self.train_dataset, seed=self.shuffle_train_seed)
        trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        return trainloader

    def testloader(self):
        testloader = DataLoader(self.test_dataset, batch_size = 64, shuffle=False, pin_memory=True)
        return testloader


class TextDataset(Dataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, prompt_format=0, prompt_type='neutral'):
        self.dataset_name = dataset_name
        self.train = True
        self.split_seed = split_seed
        self.label_seed = label_seed
        self.full_test = full_test
        

        self.train_size = train_size
        self.num_labelled = num_labelled
        self.num_labelled_test = num_labelled_test
        if not self.full_test and self.num_labelled_test == 0:
            self.num_labelled_test = self.num_labelled

        self.device = device
        self.prompt_format = prompt_format
        self.prompt_type = prompt_type

        self.text, self.targets = self.initialise_dataset_from_huggingface()
        self.num_classes = len(self.classes)
        self.split_train_test()
        print(len(self.train_text))
        print(len(self.test_text))
        self.select_labelled_data()


    def initialise_dataset_from_huggingface(self):
        if self.dataset_name == 'sst2':
            print('Using SST-2 dataset.')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            print(data.shape)
            return data.sentence.tolist(), data.label.tolist()
        elif self.dataset_name == 'cola':
            print(f'Using cola dataset')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            print(data.shape)
            return data.sentence.tolist(), data.label.tolist()
        elif self.dataset_name == 'mrpc':
            print('Using mrpc')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation']), pd.DataFrame(dataset['test'])])
            texts = [f'Sentence 1: {sent1}; Sentence 2: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            print(data.shape)
            return texts, data.label.tolist()
        elif self.dataset_name == 'rte':
            print('Using rte')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Premise: {sent1}; Hypothesis: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            print(data.shape)
            return texts, data.label.tolist()
        elif self.dataset_name == 'boolq':
            print('Using BoolQ dataset')
            dataset = load_dataset('super_glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Question: {question}\nPassage: {passage}' for question, passage in zip(data.question.tolist(), data.passage.tolist())]
            print(data.shape)
            return texts, data.label.tolist()
        elif self.dataset_name == 'trec':
            print('Using TREC dataset')
            dataset = load_dataset('trec')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            print(data.shape)
            return data.text.tolist(), data.coarse_label.tolist()
        elif self.dataset_name == 'ag_news':
            print('Using AG News dataset')
            dataset = load_dataset('ag_news')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            print(data.shape)
            return data.text.tolist(), data.label.tolist()
        elif self.dataset_name == 'snips':
            print('Using SNIPS dataset')
            dataset = load_dataset('benayas/snips')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            mapper = {
                'AddToPlaylist':            0,
                'GetWeather':               1,
                'SearchScreeningEvent':     2,
                'PlayMusic':                3,
                'SearchCreativeWork':       4,
                'RateBook':                 5,
                'BookRestaurant':           6,
            }
            data['label'] = data['category'].apply(lambda x: mapper[x])
            print(data.shape)
            return data.text.tolist(), data.label.tolist()
        elif self.dataset_name == 'db_pedia':
            print('Using DB Pedia dataset')
            dataset = load_dataset('fancyzhx/dbpedia_14')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            print(data.shape)
            return data.content.tolist(), data.label.tolist()
        else:
            raise NotImplemented('The dataset cannot be initiated!')


    def split_train_test(self, train_test_indices=None):
        if train_test_indices is None:
            old_state = torch.get_rng_state()
            torch.manual_seed(self.split_seed)
            indices = list(range(len(self.text)))
            self.train_indices, self.test_indices = train_test_split(indices, train_size=self.train_size, random_state=self.split_seed, stratify=self.targets)
            torch.set_rng_state(old_state)
        else:
            self.train_indices, self.test_indices = train_test_indices

        self.train_text = [self.text[idx] for idx in self.train_indices]
        self.train_targets = [self.targets[idx] for idx in self.train_indices]

        self.test_text = [self.text[idx] for idx in self.test_indices]
        self.test_targets = [self.targets[idx] for idx in self.test_indices] 


    def select_labelled_data(self):
        if self.num_labelled > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            to_select = max(math.ceil(self.num_labelled / self.num_classes), 2)

            targets = np.array(self.train_targets)

            texts = []
            labels = []
            train_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                train_indices.extend(inds)
                for idx in inds:
                    texts.append(self.train_text[idx])
                    labels.append(self.train_targets[idx])
            indices = torch.randperm(len(labels))
            self.train_text = [texts[idx] for idx in indices]
            self.train_targets = [labels[idx] for idx in indices]
            self.train_indices = train_indices

            print(f'Number of selected Train samples: {len(self.train_targets)}')

            torch.set_rng_state(old_state)
        
        if not self.full_test and self.num_labelled_test > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            to_select = math.ceil(self.num_labelled_test / self.num_classes)

            targets = np.array(self.test_targets)

            texts = []
            labels = []
            test_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                test_indices.extend(inds)
                for idx in inds:
                    texts.append(self.test_text[idx])
                    labels.append(self.test_targets[idx])
            indices = torch.randperm(len(labels))
            self.test_text = [texts[idx] for idx in indices]
            self.test_targets = [labels[idx] for idx in indices]
            self.test_indices = test_indices
            print(len(self.test_text))

            print(f'Number of selected Test samples: {len(self.test_targets)}')

            torch.set_rng_state(old_state)

class FineTuningDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, tokenizer=None, max_len=50, augmented_data_size=0):
        super(FineTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)
        self.tokenizer = tokenizer
        self.train = True
        self.max_len = max_len
        self.augmented_data_size = augmented_data_size

        self.n_classes = self.num_classes

        if self.augmented_data_size > 0 or self.augmented_data_size == -1:
            self.load_augmented_data()

    def load_augmented_data(self):
        augmented_data = pd.read_csv(os.path.join('data', f'{self.dataset_name}.csv'))
        new_data = []
        new_labels = []
        for idx, text in enumerate(self.train_text):
            subrows = augmented_data[augmented_data.seed == text]
            if self.augmented_data_size > 0:
                new_data.extend(subrows.text.tolist()[:self.augmented_data_size])
                new_labels.extend(subrows.label.tolist()[:self.augmented_data_size])
            else:
                new_data.extend(subrows.text.tolist())
                new_labels.extend(subrows.label.tolist())
        self.train_text.extend(new_data)
        self.train_targets.extend(new_labels)
        print(f'New size of training data is {len(self.train_text)}')

    def __len__(self):
        return len(self.train_text) if self.train else len(self.test_text)

    def __getitem__(self, index):
        text = str(self.train_text[index] if self.train else self.test_text[index])
        target = self.train_targets[index] if self.train else self.test_targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }