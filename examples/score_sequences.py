"""
Example: scoring DNA sequences with Evo 2 on Apple Silicon.
"""
import evo2_metal          # must be imported first
from evo2 import Evo2

model = Evo2('evo2_7b_base')

sequences = [
    "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT",   # E. coli gene fragment
    "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGA",   # single-nucleotide variant
]

scores = model.score_sequences(sequences)

for seq, score in zip(sequences, scores):
    print(f"{seq}  →  score: {score:.4f}")
