"""
Example: generating DNA sequences with Evo 2 on Apple Silicon.
"""
import evo2_metal          # must be imported first
from evo2 import Evo2

model = Evo2('evo2_7b_base')

output = model.generate(
    prompt_seqs=["ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT"],
    n_tokens=200,
    temperature=1.0,
    top_k=4,
)

print("Prompt :", "ATGAAAGCAATTTTCGTACTGAAAGGTTCAGGT")
print("Output :", output.sequences[0])
print(f"Score  : {output.logprobs_mean[0]:.4f}")
