
import numpy as np
import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig, AutoModelForSequenceClassification
# from transformers.models.roberta.modeling_roberta import  shift
from transformers.models.bart.modeling_bart import  shift_tokens_right

# import torch.nn.functional as F


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class Pure_labse_(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Pure_labse, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.enocder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.enocder.to('cuda')
        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):

        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.enocder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :]

        logits = self.classifier(pooled_output)

        return logits

class Pure_labse(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(Pure_labse, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name,  cache_dir=args.workspace)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config,  cache_dir=args.workspace)
        self.encoder.to('cuda')
        # layers = [nn.Linear(config.hidden_size, hidden_size), nn.ReLU(), nn.Dropout(.3),
        #           nn.Linear(hidden_size, args.lebel_dim)]

        # layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        # self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return outputs['logits']