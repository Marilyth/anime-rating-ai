from transformers.models.bert.tokenization_bert import BertTokenizer
from keras.utils import pad_sequences
from typing import List

# On Windows, execute in elevated powershell:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Maximum number of word tokens per datapoint. Tied to GPU memory load.
# Lower means higher possible batch size and lower training time.
MAX_LEN=50

tokenizer: BertTokenizer = None
def tokenize(corpus: List[str]):
    """Converts a list of strings to a list of list of BERT tokens.

    Args:
        corpus (List[str]): The list of strings to be converted.

    Returns:
        List[List[str]]: A list of list of strings, containing the BERT tokens for each text in the corpus.
    """
    global tokenizer
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized = [["[CLS]"] + tokenizer.tokenize(value)[:MAX_LEN - 1] for value in corpus]

    return tokenized

def encode(tokenized_corpus: List[List[str]]):
    """Converts a tokenized string corpus to a tokenized int corpus, using BERT vocabulary.

    Args:
        tokenized_corpus (List[List[str]]): The corpus to be converted.

    Returns:
        Dict[str, List[int]]: A dictionary containing the converted ids and attention masks for the encoded corpus.
        {"ids": List[int], "attention_masks": List[int]}
    """
    ids = [tokenizer.convert_tokens_to_ids(text) for text in tokenized_corpus]
    padded_ids = pad_sequences(ids, truncating='post', padding='post', dtype='long')
    attention_masks = []
    for text in padded_ids:
        attention_masks.append([int(item > 0) for item in text])
    return {"ids":padded_ids, "attention_masks": attention_masks}
