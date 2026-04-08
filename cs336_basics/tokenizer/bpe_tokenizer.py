"""
    The BPE tokenizer class, which can encode a text of string and decode ids to texts.
"""

from __future__ import annotations

from collections.abc import Iterable

from .pretokenization import get_pre_tokens_from_sequence
from .train_bpe import load_bpe_vocab_json, load_bpe_merges_json

class BPETokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        # add special tokens to vocab
        if special_tokens is not None:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []
        for special_token in self.special_tokens:
            special_token = special_token.encode()
            if special_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token

        # map token to id, to save time for finding a token's id
        self.token_to_id: dict[bytes, int] = {}
        for key, value in vocab.items():
            self.token_to_id[value] = key
        
        # map (token1, token2) to idx
        self.merges_to_idx: dict[tuple[bytes, bytes], int] = {}
        for idx, merge in enumerate(self.merges):
            self.merges_to_idx[merge] = idx

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
        # turn string into pretokens
        pre_tokens: list[list[bytes]] = []
        _, pre_tokens = get_pre_tokens_from_sequence(text, self.special_tokens)

        # get special_token_bytes  
        special_token_bytes = {tok.encode() for tok in self.special_tokens}

        # for every pretoken, merge it looply
        for idx, pre_token in enumerate(pre_tokens):
            if len(pre_token) == 1 and pre_token[0] in special_token_bytes:
                continue
            while(True):
                # we first find merges in this pre_token
                pairs = [(self.merges_to_idx[(pre_token[i], pre_token[i+1])], i)
                        for i in range(len(pre_token) - 1)
                        if (pre_token[i], pre_token[i+1]) in self.merges_to_idx]
                # this pre_token is small enough
                if not pairs:
                    break
                # we should find the pair(merge) whose idx in self.merges is smallest
                _, mergePos = min(pairs)
                pre_token = (
                    pre_token[:mergePos]
                    + [pre_token[mergePos] + pre_token[mergePos + 1]]
                    + pre_token[mergePos + 2 :]
                )
            pre_tokens[idx] = pre_token

        # get ids
        ids = []
        for pre_token in pre_tokens:
            for token in pre_token:
                ids.append(self.token_to_id[token])

        return ids
    
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(
        self,
        ids: list[int],
    ) -> str:
        text = b""
        for id in ids:
            if id in self.vocab.keys():
                token = self.vocab[id]
                text += token
        return text.decode(errors="replace")


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files("TinyStoriesV2_GPT4_BPE.json", "TinyStoriesV2_GPT4_BPE.json", ["<|endoftext|>"])
    ans = tokenizer.encode("hello <|endoftext|> from the tokenizer.")
    print(f"tokenized text: {ans}")
    text = tokenizer.decode(ans)
    print(f"decoded: {text}")
    print(f"original: hello <|endoftext|> from the tokenizer.")