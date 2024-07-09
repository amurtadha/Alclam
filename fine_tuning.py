

import argparse
import random
import numpy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from datasets import load_metric

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
class MyDataset(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, labels):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        labels = {label: _ for _,label in enumerate(labels)}
        data = json.load(open(fname))
        all_data=[]
        for d in tqdm(data):
            text, label = d['text'], d['label']

            data = {
                'text': text,
                'label': labels[label]
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        text = self.data[index]['text']
        label = self.data[index]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)




def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }
def _train(opt):
    opt.labels = json.load(open('/data/models/datasets/alclam/{0}/labels.json'.format(opt.dataset)))
    num_labels = len(opt.labels)

    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)
    model = AutoModelForSequenceClassification.from_pretrained(opt.pretrained_bert_name, num_labels=num_labels)

    trainset = MyDataset(opt.dataset_files['train'], tokenizer, opt.max_seq_len, opt.labels)
    testset = MyDataset(opt.dataset_files['test'], tokenizer, opt.max_seq_len, opt.labels)
    valset = MyDataset(opt.dataset_files['dev'], tokenizer, opt.max_seq_len, opt.labels)


    training_args = TrainingArguments(
        per_device_train_batch_size=opt.batch_size,
        num_train_epochs=opt.num_epoch,
        logging_dir='./logs',
        logging_steps=1000,
        save_steps=500,
        output_dir='./results',
        learning_rate=opt.learning_rate,
        overwrite_output_dir=True,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
    compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()
    test_results = trainer.evaluate(testset)
    test_results['pretrained_bert_name']= opt.baseline
    test_results['task']= opt.dataset

    with open("my_results.json", 'a+') as file:
        json.dump(test_results, file, indent=4)
    # Save the fine-tuned model
    # model_path = "./fine_tuned_model"
    # model.save_pretrained(model_path)




def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Corpus-26', type=str, help='Corpus-8,Corpus-26, ')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=3, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--lebel_dim', default=3, type=int)
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--pretrained_bert_name', default='rahbi/alclam-base-v1' , type=str, help='e.g. cuda:0')
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=65, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()




    opt.seed= random.randint(20,300)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opt.dataset_files = {
        'train': '/data/models/datasets/alclam/{0}/train.json'.format(opt.dataset),
        'test': '/data/models/datasets/alclam/{0}/test.json'.format(opt.dataset),
        'dev': '/data/models/datasets/alclam/{0}/dev.json'.format(opt.dataset)
    }

    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    _train(opt)


if __name__ == '__main__':
    main()
