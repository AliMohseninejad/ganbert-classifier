from transformers import AutoModel, AutoTokenizer, BertModel, BertConfig, BertTokenizer
from typing import *


def get_bert_model(model_name: str = "bert-base-cased") -> Tuple[BertModel, BertConfig]:
    """Get the Bert pre-trained model. The model should be configured here.

    Args:
        model_name (str): name of the pre-trained model from Huggingface
        transformers. Defaults to `"bert-base-cased"`.

    Returns:
        Tuple[BertModel, BertConfig]: A tuple of the pre-trained bert model and
        the configuration used for the model.
    """
    pass


def get_tokenizer(
    model_name: str = "bert-base-cased",
) -> Tuple[BertTokenizer, BertConfig]:
    """Get the Bert pre-trained tokenizer. The tokenizer should be configured
    here.

    Args:
        model_name (str): name of the pre-trained model from Huggingface
        transformers. Defaults to `"bert-base-cased"`.

    Returns:
        Tuple[BertTokenizer, BertConfig]: A tuple of the pre-trained bert
        tokenizer and the configuration used for the tokenizer.
    """
    pass
