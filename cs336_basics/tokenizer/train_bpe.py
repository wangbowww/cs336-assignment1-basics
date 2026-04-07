from .pretokenization_mp import pre_tokenization
import json
import os

def merge(
    vocab: dict[int, bytes],
    pre_tokens: dict[tuple[bytes], int],
    vocab_size: int,
) -> list[tuple[bytes, bytes]]:
    def _split_pre_tokens(pre_tokens: dict[tuple[bytes, ...], int]):
        counts = {}
        for token, freq in pre_tokens.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    def _update_pre_tokens(pre_tokens: dict[tuple[bytes, ...], int], best_pair: tuple[bytes, bytes]):
        new_pre_tokens = {}
        for token, freq in pre_tokens.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                    new_token.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_pre_tokens[tuple(new_token)] = new_pre_tokens.get(tuple(new_token), 0) + freq
        return new_pre_tokens
    
    merges: list[tuple[bytes, bytes]] = []
    # merge util we reach vocab_size
    while len(vocab) < vocab_size:
        # split each bytes
        counts = _split_pre_tokens(pre_tokens)
        # get max one
        assert len(counts) > 0
        best_pair, best_freq = max(counts.items(), key=lambda x: (x[1], x[0]))
        # update merges and vocab
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        # actual merge, update pre_tokens
        pre_tokens = _update_pre_tokens(pre_tokens, best_pair)
    return merges

def _init_buffer(
    pre_tokens: dict[tuple[bytes, ...], int],
    pair_freq_buffer: dict[tuple[bytes, bytes], int],
    pair_pretoken_buffer: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
):
    for pre_token, freq in pre_tokens.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            pair_freq_buffer[pair] = pair_freq_buffer.get(pair, 0) + freq
            pair_pretoken_buffer.setdefault(pair, set()).add(pre_token)

def _update_with_buffer(
    pre_tokens: dict[tuple[bytes, ...], int],
    pair_freq_buffer: dict[tuple[bytes, bytes], int],
    pair_pretoken_buffer: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
):
    assert len(pair_freq_buffer) > 0
    # get best pair from buffer
    best_pair, best_freq = max(pair_freq_buffer.items(), key=lambda x: (x[1], x[0]))
    # first update pair_pretoken_buffer for previous runs
    del_pre_tokens = []
    for pre_token in pair_pretoken_buffer[best_pair]:
        if pre_token not in pre_tokens:
            del_pre_tokens.append(pre_token)
    for del_pre_token in del_pre_tokens:
        pair_pretoken_buffer[best_pair].remove(del_pre_token)
    # update pre_tokens and buffers
    old_pair_freq = {}
    new_pair_freq = {}
    for pre_token in pair_pretoken_buffer[best_pair]:
        freq = pre_tokens[pre_token]
        new_pre_token: list[bytes] = []
        i = 0
        while i < len(pre_token):
            if i+1 < len(pre_token) and (pre_token[i], pre_token[i+1]) == best_pair:
                new_token = pre_token[i] + pre_token[i+1]
                new_pre_token.append(new_token)
                i += 2
            else:
                new_pre_token.append(pre_token[i])
                i += 1
        for i in range(len(pre_token)-1):
            old_pair = (pre_token[i], pre_token[i+1])
            old_pair_freq[old_pair] = old_pair_freq.get(old_pair, 0) + freq
        for i in range(len(new_pre_token)-1):
            new_pair = (new_pre_token[i], new_pre_token[i+1])
            new_pair_freq[new_pair] = new_pair_freq.get(new_pair, 0) + freq
            pair_pretoken_buffer.setdefault(new_pair, set()).add(tuple(new_pre_token))

        del pre_tokens[pre_token]
        pre_tokens[tuple(new_pre_token)] = freq
    
    for pair, freq in old_pair_freq.items():
        pair_freq_buffer[pair] -= freq
        if pair_freq_buffer[pair] == 0:
            del pair_freq_buffer[pair]
    for pair, freq in new_pair_freq.items():
        pair_freq_buffer[pair] = pair_freq_buffer.get(pair, 0) + freq

    return best_pair

def merge_with_buffer(
    vocab: dict[int, bytes],
    pre_tokens: dict[tuple[bytes, ...], int],
    vocab_size: int,
) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    # we buffer (pair, freq), quickly find best_pair 
    pair_freq_buffer: dict[tuple[bytes, bytes], int] = {}
    # we buffer (pair, list[pre_token]), quickly update pre_tokens
    pair_pretoken_buffer: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    _init_buffer(pre_tokens, pair_freq_buffer, pair_pretoken_buffer)
    # merge util we reach vocab_size
    while len(vocab) < vocab_size:
        # get best pair from buffer and do updates
        best_pair = _update_with_buffer(pre_tokens, pair_freq_buffer, pair_pretoken_buffer)
        # update merges and vocab
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
    return merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # get pre_tokens dict[bytes, int]
    pre_tokens = pre_tokenization(input_path, special_tokens)

    # vocab initializaiton
    vocab = {i: bytes([i]) for i in range(256)}
    for idx, special_token in enumerate(special_tokens):
        vocab[256 + idx] = special_token.encode("utf-8")

    # merge loop
    # merges = merge(vocab, pre_tokens, vocab_size)
    merges = merge_with_buffer(vocab, pre_tokens, vocab_size)
    return vocab, merges

def save_bpe_json(path: str, vocab, merges):
    payload = {
        "vocab": {str(k): v.hex() for k, v in vocab.items()},
        "merges": [[a.hex(), b.hex()] for a, b in merges],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_bpe_vocab_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in payload["vocab"].items()}
    return vocab

def load_bpe_merges_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in payload["merges"]]
    return merges
