from .pretokenization import pre_tokenization

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
    pair_freq_index_buffer: dict[tuple[bytes, bytes], list[tuple[bytes, ...]]]
):
    for pre_token, freq in pre_tokens.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            pair_freq_buffer[pair] = pair_freq_buffer.get(pair, 0) + freq
            pair_freq_index_buffer.setdefault(pair, []).append(pre_token)

def _update_pre_tokens_with_buffer(
    pre_tokens: dict[tuple[bytes, ...], int],
    pair_freq_buffer: dict[tuple[bytes, bytes], int],
    pair_freq_index_buffer: dict[tuple[bytes, bytes], list[tuple[bytes, ...]]]
):
    # some pre_tokens has been removed, but pair_freq_index_buffer[pair] = list[...pre_token...]
    # so it will occur "Key Error"
    assert len(pair_freq_buffer) > 0
    # get best pair from buffer
    best_pair, best_freq = max(pair_freq_buffer.items(), key=lambda x: (x[1], x[0]))
    # update pre_tokens with buffers
    new_pairs: dict[tuple[bytes, bytes], int] = {}
    old_pairs: dict[tuple[bytes, bytes], int] = {}
    new_pairs_index: dict[tuple[bytes, bytes], list[tuple[bytes, ...]]] = {}
    for pre_token in pair_freq_index_buffer[best_pair]:
        new_pre_token: list[bytes] = []
        i = 0
        freq = pre_tokens[pre_token]
        tmp_new_pairs: dict[tuple[bytes, bytes], int] = {}
        while i < len(pre_token):
            if i+1 < len(pre_token) and (pre_token[i], pre_token[i+1]) == best_pair:
                # we should merge pre_token[i] and pre_token[i+1]
                new_token = pre_token[i] + pre_token[i+1]
                new_pre_token.append(new_token)
                if i > 0:
                    old_pair = (pre_token[i-1], pre_token[i])
                    old_pairs[old_pair] = old_pairs.get(old_pair, 0) + freq
                    new_pair = (pre_token[i-1], new_token)
                    tmp_new_pairs[new_pair] = tmp_new_pairs.get(new_pair, 0) + freq
                if i+2 < len(pre_token):
                    old_pair = (pre_token[i+1], pre_token[i+2])
                    old_pairs[old_pair] = old_pairs.get(old_pair, 0) + freq
                    new_pair = (new_token, pre_token[i+2])
                    tmp_new_pairs[new_pair] = tmp_new_pairs.get(new_pair, 0) + freq
                i += 2
            else:
                new_pre_token.append(pre_token[i])
                i += 1
        # delete old pre_token, add new_pre_token
        del pre_tokens[pre_token]
        pre_tokens[tuple(new_pre_token)] = freq
        for new_pair, freq in tmp_new_pairs.items():
            new_pairs[new_pair] = new_pairs.get(new_pair, 0) + freq
            new_pairs_index.setdefault(new_pair, []).append(tuple(new_pre_token))

        # update buffers
    for old_pair, freq in old_pairs.items():
        pair_freq_buffer[old_pair] -= freq
    for new_pair, freq in new_pairs.items():
        pair_freq_buffer[new_pair] = pair_freq_buffer.get(new_pair, 0) + freq
    # delete best_pair
    del pair_freq_buffer[best_pair]
    del pair_freq_index_buffer[best_pair]
    return best_pair

def merge_with_buffer(
    vocab: dict[int, bytes],
    pre_tokens: dict[tuple[bytes, ...], int],
    vocab_size: int,
) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    # we buffer (pair, freq)
    pair_freq_buffer: dict[tuple[bytes, bytes], int] = {}
    # we buffer (pair, pre_tokens)
    pair_freq_index_buffer: dict[tuple[bytes, bytes], list[tuple[bytes, ...]]] = {}
    _init_buffer(pre_tokens, pair_freq_buffer, pair_freq_index_buffer)
    # merge util we reach vocab_size
    while len(vocab) < vocab_size:
        # get best pair from buffer and do updates
        best_pair = _update_pre_tokens_with_buffer(pre_tokens, pair_freq_buffer, pair_freq_index_buffer)
        # update merges and vocab
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
    return merges

def train_bpe(
    input_path: str,
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
