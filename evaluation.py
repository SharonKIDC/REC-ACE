from os import path
from typing import Callable, Dict, List
import warnings
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
import numpy as np
import pandas as pd
from tabulate import tabulate
from bert_score.scorer import BERTScorer
import transformers
import logging

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


class BERTS():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BERTS, cls).__new__(cls)
            cls.instance.scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', device=None, lang='en')
        return cls.instance

    def calc_score(self, reference, predicted):
        return self.scorer.score(predicted, reference)


def calculate_bert_score(reference, predicted):
    """
    Calculate BERTScore (https://arxiv.org/pdf/1904.09675.pdf) between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: Bert F1 score.

    """

    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]
    assert len(_reference) == len(_predicted), "The number of reference and predicted sentences must be the same."

    scorer = BERTS()
    bert_score = scorer.calc_score

    def _calc_bert(reference, predicted):
        reference_tokens = [reference]
        predicted_tokens = [predicted]

        _, _, F1 = bert_score(reference_tokens, predicted_tokens)
        return F1.mean()

    return sum([_calc_bert(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)



def calculate_wer(reference, predicted):
    """
    Calculate Word Error Rate (WER) between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: WER score.

    """
    
    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]
    assert len(_reference) == len(_predicted), "The number of reference and predicted sentences must be the same."

    def _calc_wer(reference, predicted):
        reference_tokens = reference.split()
        predicted_tokens = predicted.split()

        # Calculate Levenshtein distance
        distance = Levenshtein.distance(reference_tokens, predicted_tokens)

        # Calculate WER
        wer = distance / len(reference_tokens)
        return wer
    
    return sum([_calc_wer(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)


def calculate_exact_match(reference, predicted):
    """
    Calculate Exact Match (EM) between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: EM score.

    """
    
    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]
    assert len(_reference) == len(_predicted), "The number of reference and predicted sentences must be the same."

    def _calc_em(reference, predicted):
        return int(reference == predicted)
    
    return sum([_calc_em(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)


def calculate_bleu(reference, predicted):
    """
    Calculate BLEU score between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: BLEU score.

    """
    
    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]
    assert len(_reference) == len(_predicted), "The number of reference and predicted sentences must be the same."

    def _calc_bleu(reference, predicted):
        reference_tokens = reference.split()
        predicted_tokens = predicted.split()

        # Calculate BLEU score
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            bleu = sentence_bleu([reference_tokens], predicted_tokens)
        return bleu
    
    return sum([_calc_bleu(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)


def calculate_gleu(reference, predicted):
    """
    Calculate GLEU score between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        float: GLEU score.

    """
    
    _reference = reference if isinstance(reference, list) else [reference]
    _predicted = predicted if isinstance(predicted, list) else [predicted]
    assert len(_reference) == len(_predicted), "The number of reference and predicted sentences must be the same."

    def _calc_gleu(reference, predicted):
        reference_tokens = reference.split()
        predicted_tokens = predicted.split()

        # Calculate GLEU score
        glue = sentence_gleu([reference_tokens], predicted_tokens)
        return glue
    
    return sum([_calc_gleu(ref, pred) for ref, pred in zip(_reference, _predicted)]) / len(_reference)


def calculate_metrics(reference, predicted):
    """
    Calculate WER, EM, BLEU and GLEU scores between reference and predicted sentences.

    Args:
        reference (str or list): Reference sentence(s).
        predicted (str or list): Predicted sentence(s).

    Returns:
        tuple: WER, EM, BLEU, GLEU and BERTscore scores.

    """
    
    wer = calculate_wer(reference, predicted)
    em = calculate_exact_match(reference, predicted)
    bleu = calculate_bleu(reference, predicted)
    gleu = calculate_gleu(reference, predicted)
    bs = calculate_bert_score(reference, predicted)
    return wer, em, bleu, gleu, bs


def get_metric_func(metric_name: str) -> Callable:
    """
    Get the metric function by name.

    Args:
        metric_name (str): Name of the metric function.

    Returns:
        Callable: Metric function.

    """
    
    if metric_name == "wer":
        return calculate_wer
    elif metric_name == "em":
        return calculate_exact_match
    elif metric_name == "bleu":
        return calculate_bleu
    elif metric_name == "gleu":
        return calculate_gleu
    elif metric_name == "bs":
        return calculate_bert_score
    else:
        raise ValueError(f"Metric function '{metric_name}' does not exist.")


class Evaluator:
    def __init__(self, metrics: List[str], set_types: List[str] = ["train", "dev", "test"]):
        self.metrics = { metric_name: get_metric_func(metric_name) for metric_name in metrics}
        self.set_types = set_types

        # Create metrics dataframe with index epoch number, columns: metric1, metric2, ...
        self.metrics_df = { set_type: pd.DataFrame(columns=list(self.metrics.keys())) for set_type in self.set_types }

        self.current_epoch = 1
        self.epoch_metrics = self._initialize_epoch_metrics()

    def _initialize_epoch_metrics(self):
        return { set_type: { metric_name: [] for metric_name in self.metrics.keys() } for set_type in self.set_types }
    
    def calculate_metrics(self, set_type, reference, predicted):
        for metric_name, metric_func in self.metrics.items():
            self.epoch_metrics[set_type][metric_name].append(metric_func(reference, predicted))

    def end_epoch_routine(self, print_metrics=False, indent = 0):
        # Calculate mean of metrics for each set type
        for set_type in self.set_types:
            for metric_name in self.metrics.keys():
                self.metrics_df[set_type].loc[self.current_epoch, metric_name] = \
                    np.mean(self.epoch_metrics[set_type][metric_name])

        if print_metrics:
            self._print_epoch_metrics(self.current_epoch, indent)
        
        self.current_epoch += 1
        self.epoch_metrics = self._initialize_epoch_metrics()

    def _print_epoch_metrics(self, epoch_num, indent = 0):
        set_reports = [f"{set_type.capitalize()}: " + 
                       (', '.join([f"{metric_name}={self.metrics_df[set_type].loc[epoch_num, metric_name]:.04}" 
                                  for metric_name in self.metrics.keys()]))
                       for set_type in self.set_types]
        print('\t'*indent + "Metrics\t|\t" + " ; ".join(set_reports))
    
    def _print_metrics_table(self, epoch_num = None):
        # epoch_num is None - print all epochs
        # epoch_num is int - print only this epoch

        for set_type in self.set_types:
            if epoch_num is None:
                df = self.metrics_df[set_type]
            else:
                df = self.metrics_df[set_type].loc[epoch_num:epoch_num+1]
            
            print(f"{set_type.capitalize()} Metrics:")
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".3f"))
            print()

    def print_final_metrics(self):
        self._print_metrics_table(epoch_num=None)
    
    def store_df(self, dirpath, losses = None):
        df = self.get_merged_df(losses) if losses is not None else self.metrics_df
            
        for set_type in self.set_types:
            df[set_type].to_csv(path.join(dirpath, f"{set_type}_metrics.csv"))

    def get_merged_df(self, losses):        
        merged_df = {set_type: self.metrics_df[set_type].copy() for set_type in self.set_types}
        for set_type in self.set_types:
            merged_df[set_type]["loss"] = losses[set_type]

        return merged_df
    
    @staticmethod
    def get_best_epoch(dirpath, metric_name = "wer"):
        """
        Get the epoch number of the best epoch in the directory.

        Parameters:
            dirpath (str): Path to the directory containing the saved model states.

        Returns:
            int: The epoch number of the best epoch.
        """

        dev_metrics_file = path.join(dirpath, "dev_metrics.csv")
        assert path.exists(dev_metrics_file), f"File '{dev_metrics_file}' does not exist."

        metrics_df = pd.read_csv(dev_metrics_file, index_col=0)
        if metric_name == "em":
            best_epoch = metrics_df[metric_name].idxmax()
        else: # wer, bleu, gleu, loss
            best_epoch = metrics_df[metric_name].idxmin()
        
        return best_epoch


if __name__ == "__main__":
    reference_transcript = "The quick brown fox jumped over the lazy dog."
    predicted_transcripts = {
        "original": "The quick brown fox jumped over the lazy dog.",
        "3mistakes": "The quick broan fox jumps over the lacy dog.",
        "1deletion": "The brown fox jumped over the lazy dog.",
        "1replacement": "The brown quick fox jumped over the lazy dog.",
        "1replacement + 1mistake": "The brown quick fox jumps over the lazy dog."
    }


    for caption, calculate_metric in zip(["WER", "EM", "BLEU", "GLEU", "BertScore"],
                                         [calculate_wer, calculate_exact_match, calculate_bleu, calculate_gleu,
                                          calculate_bert_score]):
        for error_type, predicted_transcript in predicted_transcripts.items():
            print(f"{caption}:\t{calculate_metric(reference_transcript, predicted_transcript):.4f} ({error_type})")
        
        print()