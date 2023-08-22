import os
import re
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config


class RecAceEmbeddingBlock(nn.Module):
    """
    An embedding block that for our REC-ACE model, that can handle both confidence scores vector and input ids vector.

    return an embedding vector of the words and scores embeddings summed up.
    """

    def __init__(self, words_embedding: nn.Embedding, bin_size: int = 10):
        super().__init__()
        self.words_emb = words_embedding
        embedding_size = self.words_emb.embedding_dim
        self.scores_emb = nn.Embedding(bin_size + 2, embedding_size, padding_idx=0)

    def forward(self, input_ids, scores_ids):
        x_words_emb = self.words_emb(input_ids)
        x_scores_emb = self.scores_emb(scores_ids)
        return x_words_emb + x_scores_emb


class RecACEWrapModel(nn.Module):
    """
    RecACE Wrapper module.
    Can handle input for both original T5 and our Rec ACE method.
    """

    def __init__(self, t5_type, model_type, use_pretrained=False, bin_size=None):
        super().__init__()
        assert t5_type in ['t5-small'], f"{t5_type} is not a valid model type"
        assert model_type in ['original', 'rec_ace'], f"{model_type} is not a valid training scheme"
        self.model_config = T5Config.from_pretrained(t5_type)
        self.model = T5ForConditionalGeneration.from_pretrained(
            t5_type) if use_pretrained else T5ForConditionalGeneration(config = self.model_config)

        self.model_type = model_type

        self.rec_ace_block = None
        if self.model_type == 'rec_ace':
            assert bin_size is not None, f'For model {model_type}, bin size must be defined in advance'
            self.rec_ace_block = RecAceEmbeddingBlock(words_embedding=self.model.encoder.get_input_embeddings(),
                                                      bin_size=bin_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, scores_ids: torch.Tensor = None):
        """
        Forward pass for the model
        :param input_ids: words tokens
        :param labels: ground truth
        :param scores_ids: confidence scores tokens
        :return:
        """
        if self.model_type == 'original':
            results = self.model(input_ids=input_ids, labels=labels)
        else:
            assert scores_ids is not None, "Must supply scores tokens"
            assert self.rec_ace_block is not None, "Something went wrong when constructing the model"
            inputs_embeds = self.rec_ace_block(input_ids=input_ids, scores_ids=scores_ids)
            results = self.model(inputs_embeds=inputs_embeds, labels=labels)
        return results
    
    def save_to_disk(self, filepath):
        """
        Save the current state of the model to a file on disk.

        Parameters:
            filepath (str): Path to the file where the model state will be saved.
        """
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load_from_disk(cls, filepath, t5_type, model_type, use_pretrained=False, bin_size=None):
        """
        Load the model state from a file on disk and create a new instance of the class.

        Parameters:
            filepath (str): Path to the file containing the saved model state.
            t5_type (str): T5 model type.
            model_type (str): Model type ('original' or 'rec_ace').
            use_pretrained (bool): Whether to use pretrained weights for the base T5 model.
            bin_size (int): Size of the score bins (required for 'rec_ace' model).

        Returns:
            RecACEWrapModel: An instance of the class with the loaded state.
        """
        new_instance = cls(t5_type, model_type, use_pretrained, bin_size)
        new_instance.load_state_dict(torch.load(filepath))
        return new_instance


def detokenize_and_clean(tokenizer, embedded_predicted):
    """
    Detokenize the predicted sentences and clean them from <.*> tags.
    :param tokenizer: T5 tokenizer
    :param embedded_predicted: list of predicted sentences
    :return: list of cleaned sentences
    """
    # Detokenize    
    sentences = tokenizer.batch_decode(embedded_predicted)
    
    # find all <.*> and replace with ''
    sentences = [re.sub('<.*?>', '', sentence) for sentence in sentences]

    return sentences


if __name__ == '__main__':
    model_type = 'rec_ace'
    model = RecACEWrapModel(t5_type='t5-small', model_type=model_type, use_pretrained=False, bin_size=10)
    input_ids = torch.randint(low=0, high=500, size=(1, 500))
    labels = torch.randint(low=0, high=500, size=(1, 500))
    scores_ids = torch.randint(low=0, high=10, size=(1, 500))
    output = model(input_ids=input_ids, labels=labels, scores_ids=scores_ids)
    print(output.loss)
