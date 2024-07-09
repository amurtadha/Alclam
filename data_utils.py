import random

from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np

class Process_topic(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        data = open(fname)

        all_data=[]
        for line in data:
            if len(line.split())<3:continue
            inputs = tokenizer.encode_plus(line.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
class Process_Corpus(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        labels = json.load(open('/data/models/datasets/alclam/{0}/labels.json'.format(dataset)))
        data = open(fname).read().splitlines()

        all_data=[]
        for d in tqdm(data):
            text, label = d.split('\t')
            if label not in labels:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')
            # print(labels)
            # print(label)
            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
class Process_Corpus_Binary(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        fname = fname.replace('2', '26').replace('66', '6')

        # labels = json.load(open('/workspace/June/NLP_ADI/dataset/{0}/labels.json'.format(dataset)))

        data = open(fname).read().splitlines()

        all_data=[]
        for d in tqdm(data):
            text, label = d.split('\t')
            # if label not in labels:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')
            # print(labels)
            # print(label)
            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': 0 if label=='MSA' else 1
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class Process_Corpus_json(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len

        labels = {label: _ for _,label in enumerate(labels)}

        data = json.load(open(fname))

        all_data=[]
        for d in tqdm(data):
            text, label = d['text'], d['label']

            if label not in labels: continue
                # print(d)

            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'text': text,
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
            }
            all_data.append(data)
            # if len(all_data)>10000:break
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class Process_Corpus_ads(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset, train_len):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        labels = json.load(open('datasets/{0}/labels.json'.format(dataset)))
        data = open(fname)
        indexes = np.random.choice(np.arange(1000000), train_len)
        all_data=[]
        for i, text in enumerate(data):
            if i not in indexes:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels['DOH']
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
