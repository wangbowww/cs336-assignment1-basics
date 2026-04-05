from __future__ import annotations
from .pretokenization import get_pre_tokens_from_sequence
from .train_bpe import load_bpe_vocab_json, load_bpe_merges_json

class BPETokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        pass

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        vocab = load_bpe_vocab_json(vocab_filepath)
        merges = load_bpe_merges_json(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(
        self,
        text: str
    ) -> list[int]:
        return []
    
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterable[int]:
        pass

    def decode(
        self,
        ids: list[int],
    ) -> str:
        return ""