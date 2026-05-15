"""Bridge layer: Push-GP genomes ↔ LDPC decoder."""
from .adapter import make_callables, oms_seed_genome, save_oms_seed, load_oms_seed

__all__ = ["make_callables", "oms_seed_genome", "save_oms_seed", "load_oms_seed"]
