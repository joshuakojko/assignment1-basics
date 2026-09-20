import os
import regex
import heapq
from collections import defaultdict, Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class MaxPair:
    __slots__ = ("pair",)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        data = f.read()
    token_counts: Counter = Counter()
    for match in regex.finditer(PAT, data.decode("utf-8", errors="ignore")):
        token_str = match.group(0)
        if token_str:
            token_counts[token_str.encode("utf-8")] += 1

    pre_token_table: dict[int, dict] = {}
    pre_token_map: dict[bytes, int] = {}
    id_counter = 0
    for token, cnt in token_counts.items():
        pre_token_map[token] = id_counter
        pre_token_table[id_counter] = {
            "segments": [bytes([b]) for b in token],
            "count": cnt,
        }
        id_counter += 1

    def get_pairs(seq):
        return [(seq[i], seq[i+1]) for i in range(len(seq) - 1)]

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_index: dict[tuple[bytes, bytes], set] = defaultdict(set)

    for pid, row in pre_token_table.items():
        segments = row["segments"]
        count = row["count"]
        for pair in get_pairs(segments):
            pair_counts[pair] += count
            pair_index[pair].add(pid)

    initial_vocab_size = len(vocab)
    merge_budget = vocab_size - initial_vocab_size
    cur_token_id = initial_vocab_size

    if merge_budget < 0:
        raise ValueError("vocab_size must be greater than the initial vocabulary size")

    heap = [(-count, MaxPair(pair)) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    while len(merges) < merge_budget and heap:
        while heap:
            neg_count, maxpair = heapq.heappop(heap)
            best_pair = maxpair.pair
            actual_count = pair_counts.get(best_pair, 0)
            if actual_count > 0 and -neg_count == actual_count:
                break
        else:
            break
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[cur_token_id] = merged_token
        cur_token_id += 1

        affected_pids = set(pair_index[best_pair])
        for pid in affected_pids:
            row = pre_token_table[pid]
            old_segments = row["segments"]
            count = row["count"]
            i = 0
            new_segments = []
            while i < len(old_segments):
                if i < len(old_segments) - 1 and (old_segments[i], old_segments[i+1]) == best_pair:
                    new_segments.append(merged_token)
                    i += 2
                else:
                    new_segments.append(old_segments[i])
                    i += 1

            old_pairs = Counter(get_pairs(old_segments))
            new_pairs = Counter(get_pairs(new_segments))

            for pair in list(old_pairs):
                if new_pairs[pair] < old_pairs[pair]:
                    delta = old_pairs[pair] - new_pairs[pair]
                    pair_counts[pair] -= count * delta
                    if pair in pair_index and new_pairs[pair] == 0:
                        pair_index[pair].discard(pid)
                        if not pair_index[pair]:
                            del pair_index[pair]
                    if pair_counts[pair] == 0:
                        pair_counts.pop(pair, None)
                    if pair_counts.get(pair, 0) > 0:
                        heapq.heappush(heap, (-pair_counts[pair], MaxPair(pair)))
            for pair in new_pairs:
                if new_pairs[pair] > old_pairs.get(pair, 0):
                    delta = new_pairs[pair] - old_pairs.get(pair, 0)
                    pair_counts[pair] += count * delta
                    pair_index[pair].add(pid)
                    heapq.heappush(heap, (-pair_counts[pair], MaxPair(pair)))

            pre_token_table[pid]["segments"] = new_segments

        pair_counts.pop(best_pair, None)
        pair_index.pop(best_pair, None)

    return vocab, merges