import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration


class RecAceEmbeddingBlock(nn.Module):
    """
    An embedding block that for our REC-ACE model, that can handle both confidence scores vector and input ids vector.

    return an embedding vector of the words and scores embeddings summed up.
    """
    def __init__(self, vocab_size, bin_size, embedding_size):
        super().__init__()
        self.words_emb = nn.Embedding(vocab_size + 2, embedding_size, padding_idx=0)
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
    def __init__(self, t5_type, model_type, bin_size=None):
        super().__init__()
        assert t5_type in ['t5-small'], f"{t5_type} is not a valid model type"
        assert model_type in ['original', 'rec_ace'], f"{model_type} is not a valid training scheme"
        self.model = T5ForConditionalGeneration.from_pretrained(t5_type)
        self.model_type = model_type

        self.rec_ace_block = None
        if self.model_type == 'rec_ace':
            assert bin_size is not None, f'For model {model_type}, bin size must be defined in advance'
            vocab_size, embedding_size = self.model.shared.weight.shape
            self.rec_ace_block = RecAceEmbeddingBlock(vocab_size=vocab_size,
                                                      bin_size=bin_size,
                                                      embedding_size=embedding_size)

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

if __name__ == '__main__':

    model_type = 'rec_ace'
    model = RecACEWrapModel(t5_type='t5-small', model_type=model_type, bin_size=10)
    input_ids = torch.randint(low=0, high=500, size=(1, 500))
    labels = torch.randint(low=0, high=500, size=(1, 500))
    scores_ids = torch.randint(low=0, high=10, size=(1, 500))
    output = model(input_ids=input_ids, labels=labels, scores_ids=scores_ids)
    print(output.loss)

