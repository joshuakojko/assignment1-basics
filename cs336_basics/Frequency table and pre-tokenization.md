Frequency table and pre-tokenization
- Converting each pre-token into a sequence of length-one bytes objects, not ints. Iterating a bytes object yields ints, and the vocab and merges must be bytes.
- Splitting the corpus on special tokens before running the regex, so no pre-token ever straddles a document boundary and no merge can cross one.
- Special tokens must be encoded to bytes before going into the vocab, and must never be split by the pre-tokenizer.

Merge loop control
- The number of merges is the target minus 256 minus the number of special tokens. Check your loop condition, since "less than or equal" gives one extra merge.
- Stop early when no adjacent pairs remain, even if the merge budget is not spent.
- Decide what happens when the target vocab size is too small to allow any merges.

Incremental counting
- The deltas are weighted by pre-token count, not by one. Decrementing by one gives the wrong winner on the next step.
- Keep the inverted index honest when pairs disappear from a pre-token, whether by eager update or by verifying on lookup. Pick one invariant and hold it.

Toy inputs to test the update rule
- A pre-token where the winning pair overlaps with itself, like "aaa" with (a, a). Process left to right and confirm the second occurrence is handled correctly after the first is merged. The naive count for that pair in "aaa" is 2, and the number of merges that actually happen is 1.
- A pre-token with two non-adjacent occurrences, like "abab" with (a, b). The left neighbor of the second occurrence is the new token from the first.

Heap, if you use one
- The tie-break wants the lexicographically greatest pair, and a min-heap on negated count will break ties by the smallest pair unless you reverse the secondary key.
- Every dict change must be paired with a push, in one place, or the true maximum can be missing from the heap with no error.

Verification
- Assert the final vocab size matches the target and the last ID is one less than it.
- On a small corpus, assert the popped winner matches a direct scan of the dict, including tie-break, until you trust the heap logic.
- Round-trip check: joining the bytes of a pre-token's segmentation reproduces its original encoding.