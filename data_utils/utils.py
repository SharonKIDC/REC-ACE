from typing import List, Dict

import json
import numpy as np
import tqdm

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def read_json(json_path):
    with open(json_path, 'r') as fp:
        json_file = json.load(fp)
    return json_file


def get_bin_edges(num_bins):
    return np.linspace(0, 1, num_bins + 1)


def quantize_to_bins(vec, num_bins):
    bin_edges = get_bin_edges(num_bins)
    vec_as_bins = np.digitize(vec, bin_edges) - 1
    return vec_as_bins


def process_data(data: list, lowcase=True):
    data_processed = []
    for datapoint in tqdm.tqdm(data):
        new_datapoint = datapoint.copy()
        del new_datapoint['asr']

        asr = datapoint['asr']
        words, scores, right_or_wrong = zip(*asr)
        words_vec = ' '.join(words)
        if lowcase:
            words_vec = words_vec.lower()
        scores_vec = np.array(scores)
        row_vec = np.array(right_or_wrong)
        new_datapoint.update(
            {
                'words_vec': words_vec,
                'scores_vec': scores_vec,
                'row_vec': row_vec,
            }
        )
    data_processed.append(new_datapoint)
    return data_processed
