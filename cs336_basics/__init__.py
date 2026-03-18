import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .BPE_tokenizer import train_bpe
