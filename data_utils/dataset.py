import os
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

from data_utils.utils import read_json, quantize_to_bins


class SentencesDataset(Dataset):
    def __init__(self, sentences, labels, scores, errors):
        self.sentences = sentences
        self.scores = scores
        self.errors = errors
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentences = torch.LongTensor(self.sentences[idx])
        scores = torch.tensor(self.scores[idx])
        errors = torch.tensor(self.errors[idx])
        labels = torch.LongTensor(self.labels[idx])
        return {
            'sentences': sentences,
            'scores': scores,
            'errors': errors,
            'labels': labels
        }

def prepare_data_basic(data: list, tokenizer=None, batch_size=8, shuffle=True, lowcase=True, num_bins=10, debug=False):
    """
    Prepare data for training of REC-ACE and baseline T5
    :param data: list of data loaded from jsons
    :param tokenizer: what tokenizer to use
    :param batch_size: batch size for data loader
    :param shuffle: to shuffle or not to shuffle
    :param lowcase: whether to use transform all sentences in data to lower case
    :param num_bins: number of bins when quantize the coinfidence scores
    :return: DataLoader
    """
    assert tokenizer, "Must include tokenizer"

    # First enroll all data to lists, for easier encoding with tokkenizer
    truth_list = []
    sentences_list = []
    scores_list = []
    errors_list = []
    ids_list = []

    if debug:
        N_data = len(data)
        use_only = 0.1
        to_use = round(use_only * N_data)
        data = data[:to_use]
        print(f'Debug Mode - using only {to_use} out of {N_data}, training datapoints')

    for datapoint in data:
        # Prepare data for tokenizer
        asr = datapoint['asr']
        words_temp, scores, errors = zip(*asr)
        words_vec = ' '.join(words_temp)
        if lowcase:
            words_vec = words_vec.lower()

        # Need to duplicate scores and errors for sub-words in tokenizer.
        # So first thing to do is a temp tokens vec to understand what are the subwords
        tokens_vec = tokenizer.tokenize(words_vec)
        idx = 0
        scores_duplicated = []
        errors_duplicated = []
        # quantize scores into bins
        scores_binned = quantize_to_bins(scores, num_bins=num_bins)
        for token in tokens_vec:
            # Each word in T5 Tokenizer starts with '▁', if word doesn't start with this character the score and the
            # errors for that idx should be duplicated.
            if token.startswith('▁'):
                # One is added to both errors and scores because we want 0 to represent padding.
                scores_duplicated.append(scores_binned[idx] + 1)
                errors_duplicated.append(errors[idx] + 1)
                idx += 1
            else:
                scores_duplicated.append(scores_binned[idx - 1] + 1)
                errors_duplicated.append(errors[idx - 1] + 1)

        ids_list.append(datapoint['id'])
        truth_list.append(datapoint['truth'])
        sentences_list.append(words_vec)
        scores_list.append(scores_duplicated)
        errors_list.append(errors_duplicated)

    print('- Converting the input sentences into tokens')
    sentences = tokenizer(sentences_list, padding=True, return_tensors='pt')['input_ids']
    print('- Converting the GT sentences into tokens')
    labels = tokenizer(truth_list, padding=True, return_tensors='pt')['input_ids']
    pad_width = lambda vec: max(0, len(sentences[0]) - len(vec))
    scores = np.array([np.pad(score, (0, pad_width(score))) for score in scores_list])
    errors = np.array([np.pad(error, (0, pad_width(error))) for error in errors_list])

    dataset = SentencesDataset(sentences=sentences, scores=scores, errors=errors, labels=labels)
    data_sequences = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_sequences

def prepare_data_for_prompt_engineering(data: list, tokenizer=None, batch_size=8, shuffle=True, lowcase=True, num_bins=10, debug=False):
    """
    Prepare data for training T5 with prompt engineering
    :param data: list of data loaded from jsons
    :param tokenizer: what tokenizer to use
    :param batch_size: batch size for data loader
    :param shuffle: to shuffle or not to shuffle
    :param lowcase: whether to use transform all sentences in data to lower case
    :param num_bins: number of bins when quantize the coinfidence scores
    :return: DataLoader
    """
    assert tokenizer, "Must include tokenizer"
    prompt = """correct the sentence based on confidence scores: sentence: "{}",scores: "{}" """
    # First enroll all data to lists, for easier encoding with tokkenizer
    truth_list = []
    sentences_list = []
    scores_list = []
    errors_list = []
    ids_list = []

    if debug:
        N_data = len(data)
        use_only = 0.1
        to_use = round(use_only * N_data)
        data = data[:to_use]
        print(f'Debug Mode - using only {to_use} out of {N_data}, training datapoints')

    for datapoint in data:
        # Prepare data for tokenizer
        asr = datapoint['asr']
        words_temp, scores, errors = zip(*asr)
        words_vec = ' '.join(words_temp)
        if lowcase:
            words_vec = words_vec.lower()
        # quantize scores into bins
        scores_binned = quantize_to_bins(scores, num_bins=num_bins)
        ids_list.append(datapoint['id'])
        truth_list.append(datapoint['truth'])
        sentence =  prompt.format(words_vec, ', '.join(map(str, scores_binned)))
        sentences_list.append(sentence)
        scores_list.append(scores_binned)
        errors_list.append(errors)

    print('- Converting the input sentences into tokens')
    sentences = tokenizer(sentences_list, padding=True, return_tensors='pt')['input_ids']
    print('- Converting the GT sentences into tokens')
    labels = tokenizer(truth_list, padding=True, return_tensors='pt')['input_ids']
    pad_width = lambda vec: max(0, len(sentences[0]) - len(vec))
    scores = np.array([np.pad(score, (0, pad_width(score))) for score in scores_list])
    errors = np.array([np.pad(error, (0, pad_width(error))) for error in errors_list])

    dataset = SentencesDataset(sentences=sentences, scores=scores, errors=errors, labels=labels)
    data_sequences = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_sequences

if __name__ == '__main__':

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    path = '../data/default/train_clean.json'
    debug = True
    # Basic true loads data loader for REC-ACE training. Basic False load data loader for prompt engineering.
    basic = False

    prepare_data = prepare_data_basic if basic else prepare_data_for_prompt_engineering
    data = read_json(json_path=path)
    train_loader = prepare_data(data, tokenizer=tokenizer, debug=debug)
    batch = next(iter(train_loader))
    print(batch)