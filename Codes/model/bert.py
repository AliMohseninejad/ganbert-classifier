from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertConfig,
    BertTokenizer,
    AutoConfig,
)
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
    transformer = AutoModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    transformer.config.pad_token_id = transformer.config.eos_token_id
    return (transformer, config)


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    return (tokenizer, config)
