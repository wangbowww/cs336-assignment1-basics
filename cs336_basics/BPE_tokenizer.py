from .pretokenization import pre_tokenization

def _split_pre_tokens(pre_tokens: dict[tuple[bytes], int]):
    counts = {}
    for token, freq in pre_tokens.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts

def _update_pre_tokens(pre_tokens: dict[tuple[bytes], int], best_pair: tuple[bytes, bytes]):
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

def merge(
    vocab: dict[int, bytes],
    pre_tokens: dict[tuple[bytes], int],
    vocab_size: int,
) -> list[tuple[bytes, bytes]]:
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
    merges = merge(vocab, pre_tokens, vocab_size)
    return vocab, merges
