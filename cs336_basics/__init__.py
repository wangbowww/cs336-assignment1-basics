import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")


from .tokenizer import train_bpe
from .tokenizer import save_bpe_json
from .tokenizer import load_bpe_json
