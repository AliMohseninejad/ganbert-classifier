from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertConfig,
    BertTokenizer,
    AutoConfig,
)
from typing import *
from transformers.adapters.composition import Fuse

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
    return (transformer, config)


def get_bert_model_with_adapter(model_name: str = "bert-base-cased") -> BertModel:
    """Get the Bert pre-trained model. The model should be configured here.

    Args:
        model_name (str): name of the pre-trained model from Huggingface
        transformers. Defaults to `"bert-base-cased"`.

    Returns:
        Tuple[BertModel, BertConfig]: A tuple of the pre-trained bert model and
        the configuration used for the model.
    """

    transformer = BertModel.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    for param in transformer.parameters():
        param.requires_grad = False

    # Add an adapter for Named Entity Recognition (NER) task
    transformer.add_adapter("ner", config="pfeiffer", adapter_type="text_task")
    
    # Make the adapter parameters trainable
    for name, param in transformer.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
            
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
    config = AutoConfig.from_pretrained(model_name)
    return (tokenizer, config)
