"""
evo2-metal: Run ARC Institute's Evo 2 DNA language model on Apple Silicon (Metal GPU).

Usage:
    import evo2_metal          # apply patches before importing evo2
    from evo2 import Evo2

    model = Evo2('evo2_7b_base')
    output = model.generate(prompt_seqs=["ATGAAAGCAAT"], n_tokens=200)
"""

from evo2_metal.patch import apply_patches

apply_patches()
