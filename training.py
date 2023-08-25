""" This is a sample training script for T5 model """

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

from data_utils.dataset import prepare_data_basic
from data_utils.utils import read_json


if __name__ == '__main__':
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    path = 'data/default/train_clean.json'
    data = read_json(json_path=path)
    train_loader = prepare_data_basic(data, tokenizer=tokenizer)
    batch = next(iter(train_loader))

    input_ids = batch['sentences']
    labels = batch['labels']
    loss = model(input_ids=input_ids, labels=labels).loss
