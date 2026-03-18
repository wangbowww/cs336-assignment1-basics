import os
from typing import BinaryIO

import regex as re
from .constant import PAT


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# we have chunked_doc here, which is string
# use transfer to ['', '', '']
def pre_tokenization(
    path: str,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    def get_partial_pre_tokens(doc: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
        pre_tokens: dict[tuple[bytes, ...], int] = {}
        # split by special tokens
        parts = re.split("|".join(re.escape(tok) for tok in special_tokens), doc)
        for part in parts:
            for m in re.finditer(PAT, part):
                token = m.group(0).encode("utf-8")
                token = tuple(token[i:i+1] for i in range(len(token)))
                pre_tokens[token] = pre_tokens.get(token, 0) + 1
        return pre_tokens
    ## Usage
    with open(path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        all_pre_tokens: dict[tuple[bytes], int] = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pre_tokens = get_partial_pre_tokens(chunk, special_tokens)
            for key,value in pre_tokens.items():
                all_pre_tokens[key] = all_pre_tokens.get(key, 0) + value
        return all_pre_tokens