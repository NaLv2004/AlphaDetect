"""Generic Push-GP evolution loop (fitness-agnostic).

Plug-in `fitness_fn(genome) -> float` (smaller = better).  We never
import LDPC code here — that hookup happens in PR6/PR7.

Hard rules enforced here:
  * Every offspring is run through `validate_genome` before entering
    the next population.  Up to `max_retries` attempts per slot; if
    none survive, the parent is cloned instead.
  * Elitism keeps the top `elitism` individuals untouched.
  * Initial population is built from caller-supplied seeds (random
    program generation alone almost never satisfies permutation
    invariance — see PR3 design notes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crossover import crossover_genome
from .genome import Genome
from .mutation import mutate_genome
from .random_program import RandomProgramGenerator
from .selection import tournament_select
from .serialize import genome_fingerprint
from .validators import validate_genome


FitnessFn = Callable[[Genome], float]


@dataclass
class EvolutionConfig:
    pop_size: int = 50
    generations: int = 200
    elitism: int = 3
    tournament_k: int = 5
    p_crossover: float = 0.5
    n_mutations: int = 2
    p_const_tweak: float = 0.2
    max_retries: int = 30
    validator_deg: int = 8
    seed: int = 0
    # ---- from-scratch / diversity-first knobs ----
    # Maximum candidate generations per slot when filling a population
    # (initial random fill OR per-generation offspring fill).  No
    # parent.copy() fallback — if exhausted, raise.
    max_attempts_per_slot: int = 500
    # Random program length range for from-scratch initialization.
    rand_min_size: int = 4
    rand_max_size: int = 30
    # Worst-fitness individuals get more aggressive mutation; this is the
    # mutation count applied to the rank-worst individual.  Best
    # individual still gets `n_mutations`.  Linear interpolation by rank.
    n_mutations_max: int = 6
    # Reject duplicates by structural fingerprint.
    dedup: bool = True
    # If True, use C++ pybind11 seeder (pushgp_cpp_seeder.parallel_seed)
    # instead of the Python multiprocessing pipeline for the *initial*
    # random fill (and dedup top-up) of pop_v / pop_c.  Bit-identical VM
    # and validator semantics; ~7x faster on pop=100.
    cpp_seeder: bool = False

    # ---- BP-equivalence DCE pass (post-seeding + post-offspring) ----
    # When enabled and an ``dce_oracle`` is supplied to
    # ``evolve_from_scratch``, the top-level Python loop calls
    # ``pushgp.dce.reduce_populations_bp`` after the initial seeding
    # and after each generation's offspring fill (BEFORE the next
    # fitness eval).  The reducer pairs each V (or C) program with a
    # random peer from the other side and removes instructions that
    # do not change the BP post-LLR on a small reference frame bank
    # (rounded to ``dce_bp_decimals``).  Cpp acceleration is used by
    # default (``pushgp_cpp_dce.reduce_bp_batch``).
    dce_bp_enabled: bool = False
    dce_bp_max_iter: int = 8
    dce_bp_decimals: int = 6
    dce_bp_max_passes: int = 800
    dce_bp_max_decode_evals: int = -1   # <0 -> unlimited
    dce_bp_threads: int = 0             # 0 -> use ``workers`` arg
    dce_bp_use_cpp: bool = True

    # ---- Pair-binding (co-adaptation) -----------------------------
    # In the original two-pop `evolve_from_scratch`, V2C and C2V
    # populations are paired by a *fresh random permutation* every
    # generation.  This decouples the per-side searches but injects
    # large credit-assignment noise: a "good" V2C is only good with a
    # matching "good" C2V, and random pairing routinely destroys such
    # co-adapted partners.  Setting ``bind_pairs=True`` (the new
    # default) keeps (v[i], c[i], k[i]) bound from the very first
    # sampling: identity pairing, single triple tournament for parent
    # selection, atomic triple crossover, per-side mutation (which
    # preserves the slot index).  The three populations still exist
    # as parallel arrays but behave as one population of Genomes.
    # Setting ``bind_pairs=False`` restores the legacy per-side
    # offspring path with random permutation pairing.
    bind_pairs: bool = True


@dataclass
class GenLog:
    gen: int
    best_fit: float
    mean_fit: float
    median_fit: float
    n_valid: int
    elapsed_s: float


@dataclass
class EvolutionResult:
    best_genome: Genome
    best_fitness: float
    history: List[GenLog] = field(default_factory=list)
    final_population: List[Genome] = field(default_factory=list)
    final_fitnesses: List[float] = field(default_factory=list)


def _grow_population_from_seeds(
    seeds: Sequence[Genome],
    target: int,
    rng: np.random.Generator,
    rpg: RandomProgramGenerator,
    cfg: EvolutionConfig,
) -> List[Genome]:
    """Pad seed list to `target` via repeated mutation; reject invalids."""
    pop: List[Genome] = [s.copy() for s in seeds]
    if not pop:
        raise ValueError("at least one seed genome is required")
    i = 0
    while len(pop) < target:
        parent = pop[i % len(seeds)]
        for _ in range(cfg.max_retries):
            child = mutate_genome(parent, rng, rpg, n_mutations=cfg.n_mutations,
                                  p_const_tweak=cfg.p_const_tweak)
            ok, _ = validate_genome(child, rng=np.random.default_rng(int(rng.integers(0, 2**31))),
                                    deg=cfg.validator_deg)
            if ok:
                pop.append(child)
                break
        else:
            pop.append(parent.copy())
        i += 1
    return pop


def evolve(
    fitness_fn: FitnessFn,
    seeds: Sequence[Genome],
    cfg: EvolutionConfig,
    on_generation: Optional[Callable[[GenLog, List[Genome], List[float]], None]] = None,
) -> EvolutionResult:
    rng = np.random.default_rng(cfg.seed)
    rpg = RandomProgramGenerator(rng=rng)

    pop = _grow_population_from_seeds(seeds, cfg.pop_size, rng, rpg, cfg)
    fits = [float(fitness_fn(g)) for g in pop]

    history: List[GenLog] = []
    for gen_idx in range(cfg.generations):
        t0 = time.time()
        order = np.argsort(np.where(np.isfinite(fits), fits, np.inf))
        elites = [pop[int(i)].copy() for i in order[: cfg.elitism]]
        elite_fits = [fits[int(i)] for i in order[: cfg.elitism]]

        # Build next population.
        new_pop: List[Genome] = list(elites)
        new_fits: List[float] = list(elite_fits)
        n_valid = 0

        while len(new_pop) < cfg.pop_size:
            parent_a = tournament_select(pop, fits, cfg.tournament_k, rng)
            if rng.random() < cfg.p_crossover:
                parent_b = tournament_select(pop, fits, cfg.tournament_k, rng)
                child0 = crossover_genome(parent_a, parent_b, rng)
            else:
                child0 = parent_a.copy()

            child = None
            for _ in range(cfg.max_retries):
                cand = mutate_genome(child0, rng, rpg, n_mutations=cfg.n_mutations,
                                     p_const_tweak=cfg.p_const_tweak)
                ok, _ = validate_genome(
                    cand,
                    rng=np.random.default_rng(int(rng.integers(0, 2**31))),
                    deg=cfg.validator_deg,
                )
                if ok:
                    child = cand
                    n_valid += 1
                    break
            if child is None:
                child = parent_a.copy()  # fallback
            new_pop.append(child)
            new_fits.append(float(fitness_fn(child)))

        pop, fits = new_pop, new_fits

        f_arr = np.asarray(fits, dtype=np.float64)
        f_arr = np.where(np.isfinite(f_arr), f_arr, np.inf)
        log = GenLog(
            gen=gen_idx,
            best_fit=float(f_arr.min()),
            mean_fit=float(np.mean(f_arr[np.isfinite(f_arr)])) if np.isfinite(f_arr).any() else float("inf"),
            median_fit=float(np.median(f_arr)),
            n_valid=n_valid,
            elapsed_s=time.time() - t0,
        )
        history.append(log)
        if on_generation is not None:
            on_generation(log, pop, fits)

    best_idx = int(np.argmin(np.where(np.isfinite(fits), fits, np.inf)))
    return EvolutionResult(
        best_genome=pop[best_idx].copy(),
        best_fitness=float(fits[best_idx]),
        history=history,
        final_population=pop,
        final_fitnesses=fits,
    )


__all__ = [
    "EvolutionConfig", "EvolutionResult", "GenLog",
    "TwoPopGenLog", "TwoPopResult",
    "evolve", "evolve_from_scratch",
]


# ===================================================================
# From-scratch, diversity-first evolution (TWO-POP design)
# ===================================================================
#
# Hard rules (per `/memories/repo/gp-search-principles.md`):
#   1. validate_v2c / validate_c2v gate ENTRY.  No parent-copy fallback.
#   2. NO seed.  Initial population is pure random valid programs.
#   3. Elitism off by default; configurable via cfg.elitism (per side).
#   4. Dedup by structural fingerprint, per side.
#   5. Worse individuals (by rank) get more aggressive mutation.
#
# Architecture:
#   * Two independent populations — `pop_v` (V2C programs) and
#     `pop_c` (C2V programs) — each of size `cfg.pop_size`.
#   * Plus a small "constants pool" of size `cfg.pop_size`, each entry
#     being one independent log_constants vector evolved by tweak.
#   * Pairing for fitness: positional, with a per-generation random
#     permutation.  Each generation evaluates `pop_size` triples
#     (v[π(i)], c[i], k[i]).  Per-side fitness = the fitness of the
#     pair the program participated in this gen.
#
# This means programs do NOT carry private constants and the V/C/K
# searches do not directly co-adapt — exactly the principle "no
# structural prior".

from .crossover import crossover_program
from .mutation import mutate_log_constants, mutate_program
from .parallel_init import (
    DEFAULT_WORKERS,
    parallel_fill_random,
    parallel_validate_programs,
)
from .random_program import C2V_INSTR, V2C_INSTR
from .serialize import program_to_dict
from .validators import validate_c2v, validate_v2c

# Per-program fingerprint (lighter than full genome_fingerprint —
# excludes constants since constants live in a separate pool).
def _prog_fingerprint(prog) -> str:
    import json as _json
    return _json.dumps(program_to_dict(prog), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Behavioral fingerprint.
#
# Motivation: empirical audit on the from-scratch run showed that 70%+ of the
# population had IDENTICAL V2C output vectors on a fixed input panel even
# though every program had a unique structural (syntactic) fingerprint.  This
# means most of the diversity introduced by mutation/crossover is "dead code"
# that does not affect the stack-top output, and selection sees a much smaller
# effective pool than `pop_size`.
#
# The behavioral fingerprint runs the candidate program on a small fixed panel
# of input contexts and quantizes the resulting outputs into a string.  Two
# candidates that produce the same output on every panel entry are treated as
# duplicates (they will receive identical fitness anyway, modulo evaluator
# noise, so deduplicating them is strictly beneficial for population
# diversity).  This subsumes static dead-code-elimination for the purposes of
# dedup: any DCE-equivalent pair of programs collides.
#
# The panel uses a fixed RNG seed so the fingerprint is stable across calls
# and across processes.
_BEHAV_DEG: int = 8  # matches DEFAULT_DEG in validators.py
_BEHAV_PANEL_SIZE: int = 32
_BEHAV_QUANT: int = 8  # significant figures kept (matches C++ %.8g format)

# 32-entry behavioral panel: byte-identical between Python and the C++
# seeder (cpp_seeder/src/behav_panel.hpp).  Generated once by
# cpp_seeder/src/gen_behav_panel.py — do NOT regenerate without also
# updating the C++ literals.  Includes 3 fixed extreme rows (zeros,
# ±1 alternating, ±5 alternating) so degenerate "constant-output" /
# "identity" programs are distinguished, plus 29 random rows from the
# fixed seed 0xBE4AC1D for broad coverage of input space.
_BEHAV_PANEL_V2C_LV = np.array([
    0.0,
    1.0,
    -1.0,
    -4.429313121490904,
    3.7180681225643593,
    -1.8422510099813563,
    -2.944240111774201,
    2.978053891128174,
    3.145892309156789,
    1.9608548733364213,
    0.6036942562978478,
    -2.3775784129892843,
    -2.500005473373532,
    2.6246615516833662,
    -0.3539625139596625,
    3.597270361983945,
    -0.05242604149925789,
    1.1373780568991512,
    1.5415666842737874,
    2.864876690207292,
    4.2008138972758395,
    -2.69617102214713,
    1.9317684238743071,
    0.2428401682290362,
    -1.704661996906017,
    2.441235225475708,
    -0.9612759606610952,
    -0.04324149007083378,
    2.5828596115264704,
    1.8516089468808872,
    3.474604176889155,
    2.5876566095352604,
], dtype=np.float64)
_BEHAV_PANEL_V2C_INC = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    [5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
    [0.6573715864633813, -2.8697900669971674, -0.6660533867254497, -3.9372247433374716, 2.4430036450052715, -1.5331464973900335, -0.36076551353060005],
    [-5.790515963874704, 4.867168205370163, 2.1455543786937596, 2.842902646594064, 4.759032909447409, 2.8520002027393883, -1.509388485853436],
    [1.0840503274328306, -1.5228966019738888, 0.9909322642203069, -2.9777554048676476, -1.6831563563344272, 4.782692140075424, 5.735849311988561],
    [-3.8750714784056752, 2.5594133947531006, -0.20184750646504845, -0.9550211120953334, 5.138110304635452, -3.5409354816337655, -2.1989643321501107],
    [1.7490196956523132, 1.9863333349389958, -5.531669297588129, -4.854183184752477, -1.7820030053478373, 0.3695231384912816, -0.715900608659715],
    [-0.14169408186802102, 1.3236538493346428, -4.368151562359232, -4.531618072323482, -1.7757219561735482, -1.1111456233823507, 1.045374096093088],
    [-4.557764189274966, 5.854965105315712, -0.534024250499602, 3.6194092720027786, -4.64964403888508, -3.751244724387965, 1.9459940909104958],
    [5.562416982192257, 4.624521909537632, 2.3138764447769447, 5.709375584617646, -1.3055043165674602, -0.064691827437235, 0.5996209110108701],
    [-3.6084833012311486, -4.836862940277552, 0.4619417218776558, 1.1991570930214888, -4.49590049570927, -2.8619479914886603, 3.3417749592095767],
    [1.5028175744015542, -1.4052149187384142, -5.115246476524565, 5.697827228548862, -5.0515086655724035, -0.6452829222919378, 4.0651830068626165],
    [-0.8537427389187116, 4.526038961768361, -4.489258674699138, 5.295410057205586, 1.8577300849540936, 4.30551208361366, -2.2853655679716502],
    [-1.785882954968633, 0.40774282316146593, 5.527772399955381, -2.915846098121977, -2.4060178923712794, 3.6463818664324137, -5.61261711904687],
    [-5.120838856633525, -2.421740163122876, 1.4281967598247043, -4.007630301650638, 3.8360343572951265, -5.229259710844429, -1.6769583600555347],
    [-3.0351351245632534, 5.714411544425607, 3.9856261877523735, 4.6727726760680905, 5.062428131334183, -5.672915973710234, -3.913738332597487],
    [3.7823697774044156, 5.679395498560641, -3.1937163813408005, 4.046717720688809, -2.539579877289521, -5.444010886703005, 1.8478847381447476],
    [1.5756545545211793, 2.463233730810259, 3.7956877047368778, -3.4227500757758373, 5.740233275271514, -1.0423065915284173, -5.456168496336254],
    [0.3597014630319739, 4.073027171859195, 2.972539402548019, 2.708809841645106, -4.149753148865383, 2.6956218028237124, -3.3715997425672293],
    [0.35551152758383964, 5.73101482732193, 0.7445064221856974, 2.9041075859259156, -4.124538030949289, 0.9348535760499423, -2.52966106498698],
    [-0.3544633387971299, 3.0778838821170638, -1.2337055456217492, 5.278411738218821, 4.41477406731307, 0.07765703719653061, -2.1608404701921753],
    [-5.517214902971641, 4.945874501441862, -3.3223829544590457, -2.8997778753441477, -0.020336307901036577, 2.3827540268039886, 4.083327050192548],
    [0.41907001507798913, -1.2372928461618766, -1.6522643650657827, -2.6188195251475648, 2.997539695016682, -5.686068203836747, 4.817429907917713],
    [-0.5835572161254934, -4.163574753210925, -3.371921438471627, -3.3817709021662887, -0.08906687765763621, -4.858840966409547, -5.7311923282860375],
    [-3.2879042387276285, 1.6221213158549812, 3.9252647614620066, -4.13681832883501, 0.1441993000653703, 2.644914371833117, 3.2346025322058196],
    [0.379417433838551, -0.44350328941163397, -1.7996129828841454, 1.5441741351210876, -2.9515467845997883, -4.988373300026565, -2.74280634160269],
    [1.27168351871385, 4.36687413973479, 3.27625104569327, 4.2257460213300195, 2.7196773922120894, 1.526680984486159, -2.866812502895207],
    [0.5965931745006632, -2.614211460447913, -4.574865293768197, 0.8230351363159301, 2.3273984640710452, -0.24506132961844074, -5.509028732749912],
    [3.7468736875367714, 1.2363338304767808, -4.204427397878462, -1.3551720953268704, 3.9558548960378666, -2.056239957578574, 4.388991173187101],
    [-0.48420680653059023, 2.768686124090854, 4.438119279391639, -3.8294161950982284, 3.497225739458621, 1.2042256895977719, 2.1042338624001875],
    [1.5049786716371152, 4.063452842796561, 3.3000884612194454, 4.214906894577105, -1.0151343277732447, -3.2237068437863083, 3.0859013004621776],
], dtype=np.float64)
_BEHAV_PANEL_C2V_INC = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
    [5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
    [-0.5178391694904727, 2.6913881454585997, 0.6893965279367933, 1.8281851833757727, -0.8286013560109362, -3.1867069257173304, -5.659097076536992],
    [-2.2657646788530417, 3.657798998958304, 4.793935410621382, 5.41678756463962, 2.3913063735770557, -0.858965243077499, 5.2079506644808795],
    [2.9751149388714584, 1.5776561332016588, 2.1568546868343006, 0.08020140273960052, 1.4798988147074716, 4.176045020570477, -2.886011021086163],
    [-2.91748985992837, -5.7382017385365245, 2.9473126211677005, -1.5669217350215998, -4.150815832577168, 3.2992143175245587, -0.238426789236577],
    [-0.9338151276708953, -3.5332344039154537, -4.391593076967116, 0.8101652389000176, -1.994976120639655, -5.926272239280312, 4.150729089567571],
    [1.9808704229933163, 0.2643261754245527, -4.825552122183705, 0.0841683219304219, 3.3286810213852807, -1.1534926909447805, 2.931813737457487],
    [-0.8511075407210118, -4.870494812703325, -0.6348855933444337, -4.995336057618448, -1.380612435908688, -5.182078158842726, 0.6968894880019434],
    [4.3145743723294245, 2.0470981308806078, 2.1013318818008386, -4.762539020061556, -3.336613717983235, 4.153211699100444, 3.6420859929392613],
    [-4.590389553559623, 2.0394790694098432, -3.604324191834535, 3.955765686267668, 1.111091600300922, -1.7982848563057772, 3.240892593521277],
    [-1.9190076857246865, 4.110578931262424, -3.0476009828043686, 4.989748960731244, -2.8554212456006853, 2.9052857062936503, 5.53164622283864],
    [-5.55094171215444, 5.372221886334325, -4.231011778255343, 3.7385955833503175, 5.78503825955616, 2.8221214567011863, -3.016116825953063],
    [3.914246162994882, -5.352851664381519, -0.559939803132802, 3.651151345826394, 3.3356769204751, -1.830899742528734, 3.511400758149014],
    [-1.0191216810629182, 1.4044752091800996, -1.3093944005338196, -0.5163421509582342, 1.6770875557258211, 5.7550687632028925, -4.808663493033571],
    [4.704934094292295, -5.534425249149857, -1.8688297201402726, -1.0151035625366172, -2.081630206616744, 2.8741610023813386, 1.64981416274912],
    [1.6996194626745584, 0.17423400627945362, 4.4420137621018085, 4.163860752936166, 3.114887039758699, 0.05568224456201332, -5.7931983930320285],
    [-2.9967166750083765, 5.906840038092755, 4.31914735363978, -4.716901486588957, -0.5487429334980769, 3.898920777404431, 4.764724736819014],
    [3.204537599460263, -1.4239541803443148, -2.875866209879818, -4.190555208189014, 4.931398679363552, 0.7608873158170137, -0.6186071967531612],
    [-2.9868896710251427, 1.2533370357104117, -3.98694378788222, 1.931652623017742, -2.2739609076757064, -0.5830534249185018, -4.8718994773341215],
    [-1.9252928353812608, 2.285333305432083, 0.16083672272997163, -1.7512888715496437, -0.2801949051451942, 2.9851671521625036, 1.2529905763150788],
    [1.5457489979434058, -4.778492803209793, 1.9672658823795963, -1.810282271577912, 1.6709706059741558, -5.394573925817644, 0.2920568626719495],
    [-3.940245996938802, -4.788982909538069, -5.56608152510131, 5.6261735853315855, 4.039467612474867, 4.3113416940164, 5.713212584381036],
    [-5.701369577818141, 1.1075264387288506, -5.52942344931447, 4.5990623612992305, -2.3628905365384285, 2.077210082263157, -3.2485904981588107],
    [-4.912955788828915, 4.5900785244212585, 1.4413897852105508, -3.9076848604871612, 2.088119707499539, 4.0372407385212625, 1.5048524316762455],
    [0.7588004165771904, -3.4982231078314348, -5.554842939024305, -5.428117915426038, 5.625620654253336, -3.175565791465958, 0.7902118516941243],
    [5.984537659161534, 3.3919304601487, 0.7757194106390131, 4.2798925306709314, 3.9108618382202565, 0.9056330208936281, -0.40474004461490676],
    [0.1222039253228715, 5.821370913488126, 5.566756747166259, 3.9065556716001772, 2.8973171899188657, -4.755408419082743, 1.277634566022611],
    [2.6784078835510456, 4.531876292056296, -4.166854180844967, -1.287652943466524, -5.766261504036334, -2.162266543906359, -2.481131992693572],
    [4.9952921436276245, 0.00025253092035626423, 5.7662358312432485, 0.8568771807567872, -1.8513919399745822, -0.40654511870165244, -1.6224062437682036],
    [-2.1062936729699313, 2.2218461154579785, -2.149205948248634, -4.047911888747918, -0.24109849608043454, 3.03649607205314, 4.25327242832498],
], dtype=np.float64)
_BEHAV_DEFAULT_K: np.ndarray = np.array(
    [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64
)


def _behav_fingerprint(side: str, prog,
                        evo_consts: Optional[np.ndarray] = None,
                        iter_idx: int = 0) -> str:
    """Quantized output vector on the fixed 32-entry behavioral panel.

    Bit-identical between Python and the C++ seeder (same panel, same
    ``%.8g`` formatting, same ``|`` separator) **when called with
    default arguments** (``evo_consts=None``, ``iter_idx=0``).

    Parameters
    ----------
    evo_consts : np.ndarray, optional
        Override the evolved-constant array fed to the VM.  Defaults to
        ``_BEHAV_DEFAULT_K`` (the fixed dedup panel).  Pass the
        genome's actual K values (``10 ** genome.log_constants``)
        when using this fingerprint as a DCE oracle so that instructions
        that reference ``ctx_evo_constants`` are not incorrectly pruned.
    iter_idx : int
        Value of ``ctx_iter`` fed to the VM.  Defaults to ``0`` for
        backward compatibility with seeding-time dedup.  DCE callers
        should sample multiple values (e.g., ``{0, 2, 4}``) to catch
        instructions that are guarded by iteration count.
    """
    from .validators import _make_vm, _seed_v2c_stacks, _seed_c2v_stacks  # noqa: E402

    panel_k = _BEHAV_DEFAULT_K if evo_consts is None else evo_consts
    outs: List[str] = []
    finite_count = 0
    for i in range(_BEHAV_PANEL_SIZE):
        if side == "v2c":
            L_v = float(_BEHAV_PANEL_V2C_LV[i])
            incoming = _BEHAV_PANEL_V2C_INC[i]
            vm = _make_vm(incoming, channel_llr=L_v, deg=_BEHAV_DEG,
                          iter_idx=iter_idx, evo_consts=panel_k)
            _seed_v2c_stacks(vm)
        else:
            incoming = _BEHAV_PANEL_C2V_INC[i]
            vm = _make_vm(incoming, has_channel_llr=False, deg=_BEHAV_DEG,
                          iter_idx=iter_idx, evo_consts=panel_k)
            _seed_c2v_stacks(vm)
        try:
            out = vm.run(prog)
        except Exception:
            out = None
        if out is None or not np.isfinite(out):
            outs.append("nan")
        else:
            finite_count += 1
            outs.append(format(float(out), f".{_BEHAV_QUANT}g"))
    if finite_count == 0:
        return "NAN"
    return "|".join(outs)



def _const_fingerprint(k: np.ndarray, *, quant: int = 2) -> str:
    return ",".join(f"{round(float(x), quant)}" for x in k)


def _rank_to_n_mutations(rank: int, n_total: int, cfg: "EvolutionConfig") -> int:
    if n_total <= 1 or cfg.n_mutations_max <= cfg.n_mutations:
        return cfg.n_mutations
    frac = rank / float(n_total - 1)
    return int(round(cfg.n_mutations + frac * (cfg.n_mutations_max - cfg.n_mutations)))


def _validate_one(side: str, prog, *, deg: int, seed: int) -> bool:
    val = validate_v2c if side == "v2c" else validate_c2v
    ok, _ = val(prog, rng=np.random.default_rng(seed & 0xFFFFFFFF), deg=deg)
    return bool(ok)


def _evolve_side_offspring(
    *,
    side: str,
    pop: List,
    fits: List[float],
    cfg: "EvolutionConfig",
    rng: np.random.Generator,
    rpg: RandomProgramGenerator,
    instr_set: Sequence[str],
    pool=None,
    n_workers: int = 1,
    seen_fps_in: Optional[set] = None,
) -> Tuple[List, int, int]:
    """Build the next-generation pop for one side (V2C or C2V).

    Returns `(new_pop, n_attempts, n_invalid_rejected)`.

    `seen_fps_in` (optional) is a set of behavioral fingerprints already
    present in the population (typically built from `current pop` by the
    caller).  Any newly-generated offspring whose fingerprint is in this
    set is rejected — guaranteeing the next-gen pop has no duplicate of
    any existing member.  Accepted offspring also extend the set so that
    siblings within the same gen do not collide.

    Strategy:
      1. Optional elitism: copy top cfg.elitism untouched (elites are
         in the current pop by definition, so seen_fps_in already
         covers them).
      2. Round-robin: generate `pop_size - elites` candidates from
         tournament parents (mut/crossover, rank-scaled mutation).
      3. Batch-validate via multiprocessing pool.
      4. Accept valid + non-colliding.  Repeat until pop full.
    """
    order = np.argsort(np.where(np.isfinite(fits), fits, np.inf))
    rank_of: Dict[int, int] = {int(idx): r for r, idx in enumerate(order)}

    new_pop: List = []
    # Inherit the caller's "already in population" set so offspring
    # can't duplicate any current member.  Local additions track
    # newly-accepted siblings.
    seen: set = set(seen_fps_in) if seen_fps_in is not None else set()
    if cfg.elitism > 0:
        from .program import deep_copy_program
        for i in order[: cfg.elitism]:
            g = pop[int(i)]
            new_pop.append(deep_copy_program(g))

    n_attempts = 0
    n_invalid = 0
    target = cfg.pop_size
    max_attempts = cfg.max_attempts_per_slot * target

    from .program import deep_copy_program
    while len(new_pop) < target:
        if n_attempts >= max_attempts:
            raise RuntimeError(
                f"evolve gen offspring fill exhausted ({side}): "
                f"{len(new_pop)}/{target} after {n_attempts} attempts"
            )
        # Build a batch of candidate offspring.
        batch_size = max(1, min(n_workers * 8, target - len(new_pop) + 16))
        cands = []
        for _ in range(batch_size):
            parent_a = tournament_select(pop, fits, cfg.tournament_k, rng)
            n_mut = _rank_to_n_mutations(rank_of.get(_id_in(pop, parent_a), 0),
                                         len(pop), cfg)
            if rng.random() < cfg.p_crossover:
                parent_b = tournament_select(pop, fits, cfg.tournament_k, rng)
                child0 = crossover_program(parent_a, parent_b, rng,
                                           instr_set=instr_set)
            else:
                child0 = deep_copy_program(parent_a)
            cand = mutate_program(child0, rng, rpg, instr_set,
                                  n_mutations=max(1, n_mut))
            cands.append(cand)
        # Batch-validate.
        if n_workers > 1 and pool is not None:
            base_seed = int(rng.integers(0, 2**31))
            ok_list = parallel_validate_programs(
                side, cands, workers=n_workers,
                deg=cfg.validator_deg, base_seed=base_seed, pool=pool,
            )
        else:
            ok_list = [
                _validate_one(side, c, deg=cfg.validator_deg,
                              seed=int(rng.integers(0, 2**31)))
                for c in cands
            ]
        n_attempts += len(cands)
        for ok, cand in zip(ok_list, cands):
            if not ok:
                n_invalid += 1
                continue
            if cfg.dedup:
                fp = _behav_fingerprint(side, cand)
                if fp in seen:
                    continue
                seen.add(fp)
            new_pop.append(cand)
            if len(new_pop) >= target:
                break
    return new_pop, n_attempts, n_invalid


def _id_in(lst, item) -> int:
    """Identity-based lookup (tournament returns actual elements)."""
    for i, x in enumerate(lst):
        if x is item:
            return i
    return 0


# ----------------------------------------------------------- log type for two-pop


@dataclass
class TwoPopGenLog:
    gen: int
    best_fit: float
    mean_fit: float
    median_fit: float
    elapsed_s: float
    # New: per-side offspring stats
    v_attempts: int = 0
    v_invalid: int = 0
    c_attempts: int = 0
    c_invalid: int = 0


@dataclass
class TwoPopResult:
    best_genome: Genome
    best_fitness: float
    pop_v: List
    pop_c: List
    pop_k: List
    pair_perm: List[int]  # final-gen pairing
    fits: List[float]     # per-pair fitness, indexed by C-pop position
    history: List[TwoPopGenLog] = field(default_factory=list)


# ----------------------------------------------------------- main entry


def evolve_from_scratch(
    fitness_pair_fn: Callable[[Genome], float],
    cfg: "EvolutionConfig",
    *,
    workers: int = DEFAULT_WORKERS,
    batch_eval_fn: Optional[Callable[[List, List, List, List[int]], List[float]]] = None,
    on_generation: Optional[Callable[
        [TwoPopGenLog, List, List, List, List[int], List[float]], None
    ]] = None,
    dce_oracle: Optional[Dict] = None,
) -> TwoPopResult:
    """Two-pop from-scratch evolution.

    `fitness_pair_fn(Genome) -> float` evaluates a single (V2C, C2V,
    log_consts) triple, smaller = better.

    Workflow:
      Init: parallel-fill `pop_v`, `pop_c` (size pop_size each); random
            `pop_k` of log_constants.
      For each gen:
        a. Random pairing permutation π over [0, pop_size).
        b. Build genomes (v[π(i)], c[i], k[i]) and call fitness_pair_fn.
        c. Per-side fitness = pair fitness.
        d. Per-side offspring fill via `_evolve_side_offspring`
           (validation gates entry, dedup on, rank-scaled mutation).
        e. Constants pool: pure tweak with prob `p_const_tweak`,
           dedup by quantized fingerprint.
    """
    if cfg.pop_size <= 0:
        raise ValueError("pop_size must be > 0")

    rng = np.random.default_rng(cfg.seed)
    rpg = RandomProgramGenerator(rng=rng)

    # ---- Resolve DCE-BP config (top-level Python orchestration) -----
    _dce_enabled = bool(cfg.dce_bp_enabled and dce_oracle is not None
                        and "par" in dce_oracle and "rx_llrs" in dce_oracle)
    if cfg.dce_bp_enabled and dce_oracle is None:
        print("[dce_bp] cfg.dce_bp_enabled=True but no dce_oracle "
              "supplied; DCE pass disabled.", flush=True)
    _dce_threads = cfg.dce_bp_threads if cfg.dce_bp_threads > 0 else max(1, workers)
    from .program import program_length as _plen

    def _apply_dce_bp(pop_v, pop_c, pop_k, *, tag: str):
        """Run BP-equivalence DCE on the whole two-pop; replace in place."""
        if not _dce_enabled:
            return pop_v, pop_c
        from .dce import reduce_populations_bp
        t0 = time.time()
        sz_v0 = [_plen(p) for p in pop_v]
        sz_c0 = [_plen(p) for p in pop_c]

        def _cb(side, done, total, elapsed):
            if done == total or (done % max(1, total // 10) == 0):
                print(f"[dce_bp {tag}] {done}/{total} jobs done in "
                      f"{elapsed:.1f}s", flush=True)
        new_v, new_c, st_v, st_c = reduce_populations_bp(
            pop_v, pop_c, pop_k,
            par=dce_oracle["par"],
            rx_llrs=dce_oracle["rx_llrs"],
            max_iter=cfg.dce_bp_max_iter,
            max_passes=cfg.dce_bp_max_passes,
            max_decode_evals=cfg.dce_bp_max_decode_evals,
            decimals=cfg.dce_bp_decimals,
            threads=_dce_threads,
            use_cpp=cfg.dce_bp_use_cpp,
            rng=np.random.default_rng((cfg.seed * 991
                                       + (hash(tag) & 0xFFFFFFFF)) & 0xFFFFFFFF),
            on_progress=_cb,
        )
        sz_v1 = [_plen(p) for p in new_v]
        sz_c1 = [_plen(p) for p in new_c]
        rem_v = sum(sz_v0) - sum(sz_v1)
        rem_c = sum(sz_c0) - sum(sz_c1)
        print(f"[dce_bp {tag}] V removed {rem_v} instr "
              f"({float(np.mean(sz_v0)):.1f} -> {float(np.mean(sz_v1)):.1f} avg)  "
              f"C removed {rem_c} instr "
              f"({float(np.mean(sz_c0)):.1f} -> {float(np.mean(sz_c1)):.1f} avg)  "
              f"elapsed={time.time()-t0:.1f}s  "
              f"threads={_dce_threads}  use_cpp={cfg.dce_bp_use_cpp}",
              flush=True)
        return new_v, new_c

    # Choose seeding backend (Python multiprocessing or C++ pybind11).
    if cfg.cpp_seeder:
        from .cpp_seeder_adapter import cpp_parallel_fill_random as _fill_random
        print("[init] using C++ seeder (pushgp_cpp_seeder.parallel_seed)", flush=True)
    else:
        _fill_random = parallel_fill_random

    # Persistent pool reused across gens for both init and offspring
    # validation (avoids spawning overhead).
    from multiprocessing import Pool
    pool = Pool(processes=workers) if workers > 1 else None
    try:
        # ---- 1. Initial pops via parallel brute-force fill -------------
        t_init = time.time()

        def _seed_progress(side, n_valid, n_attempts, elapsed_s):
            target = cfg.pop_size
            rate = (n_valid / elapsed_s) if elapsed_s > 0 else 0.0
            remaining = max(0, target - n_valid)
            eta = (remaining / rate) if rate > 0 else float("inf")
            pass_rate = (n_valid / n_attempts) if n_attempts > 0 else 0.0
            eta_s = f"{eta:6.1f}s" if eta != float("inf") else "  inf "
            print(
                f"[init-seed {side}] {n_valid:4d}/{target}  "
                f"attempts={n_attempts:>9d}  pass={pass_rate:7.4%}  "
                f"elapsed={elapsed_s:6.1f}s  rate={rate:6.2f}/s  ETA={eta_s}",
                flush=True,
            )

        # Behavioral-fingerprint dedup is now part of validation: the
        # seeder rejects any candidate whose 32-entry behavior matches
        # any program already in `seen_v` / `seen_c`.  No post-hoc
        # dedup loop is needed.
        seen_v: set = set()
        seen_c: set = set()

        pop_v, v_attempts = _fill_random(
            "v2c", cfg.pop_size,
            max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 1000,
            workers=workers,
            chunk_attempts=max(1000, cfg.max_attempts_per_slot),
            min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
            deg=cfg.validator_deg, base_seed=cfg.seed * 17 + 1,
            pool=pool,
            progress_cb=_seed_progress,
            seen_fingerprints=seen_v,
        )
        pop_c, c_attempts = _fill_random(
            "c2v", cfg.pop_size,
            max_attempts=cfg.max_attempts_per_slot * cfg.pop_size * 1000,
            workers=workers,
            chunk_attempts=max(1000, cfg.max_attempts_per_slot),
            min_size=cfg.rand_min_size, max_size=cfg.rand_max_size,
            deg=cfg.validator_deg, base_seed=cfg.seed * 17 + 2,
            pool=pool,
            progress_cb=_seed_progress,
            seen_fingerprints=seen_c,
        )
        pop_k = [rpg.random_log_constants() for _ in range(cfg.pop_size)]

        print(f"[init] V pop filled: {len(pop_v)} valid in {v_attempts} attempts "
              f"({len(pop_v)/v_attempts:.4%})", flush=True)
        print(f"[init] C pop filled: {len(pop_c)} valid in {c_attempts} attempts "
              f"({len(pop_c)/c_attempts:.4%})", flush=True)
        print(f"[init] elapsed: {time.time()-t_init:.1f}s", flush=True)

        # ---- 1b. (Optional) BP-equivalence DCE on seeded populations.
        pop_v, pop_c = _apply_dce_bp(pop_v, pop_c, pop_k, tag="init")

        # ---- 2. Initial fitness evaluation via positional pairing -----
        # When `bind_pairs` is True, identity permutation is used so
        # (v[i], c[i], k[i]) are bound from gen 0 and reproduce as a
        # single triple unit.  Random pairing (legacy two-pop CCEA)
        # remains available via `cfg.bind_pairs=False`.
        if cfg.bind_pairs:
            perm = list(range(cfg.pop_size))
        else:
            perm = list(rng.permutation(cfg.pop_size))
        if batch_eval_fn is not None:
            t_e = time.time()
            fits = batch_eval_fn(pop_v, pop_c, pop_k, perm)
            print(f"[init-eval] {len(fits)} pairs in {time.time()-t_e:.1f}s "
                  f"(parallel)  best={min(fits):+.4f} med={float(np.median(fits)):+.4f}",
                  flush=True)
        else:
            fits = _eval_pairs(pop_v, pop_c, pop_k, perm, fitness_pair_fn,
                               prefix="[init-eval]")

        history: List[TwoPopGenLog] = []

        # ---- 3. Generation loop ---------------------------------------
        for gen_idx in range(cfg.generations):
            t0 = time.time()
            # Per-side fitness: each entry got one pair fitness this gen.
            # fits[i] is the fitness of pair (v[perm[i]], c[i], k[i]).
            fits_c = list(fits)
            fits_v = [0.0] * cfg.pop_size
            fits_k = [0.0] * cfg.pop_size
            for i, vi in enumerate(perm):
                fits_v[vi] = fits[i]
                fits_k[i] = fits[i]

            if cfg.bind_pairs:
                # Atomic triple offspring: one tournament selects the
                # parent triple index, both sides' offspring built from
                # the *same* parent indices and stored at the same
                # output slot.  Binding is preserved across generations.
                (new_pop_v, new_pop_c, new_pop_k,
                 va, vinv, ca, cinv) = _evolve_triple_offspring(
                    pop_v=pop_v, pop_c=pop_c, pop_k=pop_k,
                    fits=list(fits),  # pair fitness (identity perm)
                    cfg=cfg, rng=rng, rpg=rpg,
                    pool=pool, n_workers=workers,
                )
            else:
                new_pop_v, va, vinv = _evolve_side_offspring(
                    side="v2c", pop=pop_v, fits=fits_v, cfg=cfg, rng=rng,
                    rpg=rpg, instr_set=V2C_INSTR, pool=pool, n_workers=workers,
                    seen_fps_in={_behav_fingerprint("v2c", p) for p in pop_v},
                )
                new_pop_c, ca, cinv = _evolve_side_offspring(
                    side="c2v", pop=pop_c, fits=fits_c, cfg=cfg, rng=rng,
                    rpg=rpg, instr_set=C2V_INSTR, pool=pool, n_workers=workers,
                    seen_fps_in={_behav_fingerprint("c2v", p) for p in pop_c},
                )
                new_pop_k = _evolve_constants(
                    pop_k, fits_k, cfg, rng,
                )

            pop_v, pop_c, pop_k = new_pop_v, new_pop_c, new_pop_k

            # ---- 3b. (Optional) BP-equivalence DCE on offspring pops.
            pop_v, pop_c = _apply_dce_bp(pop_v, pop_c, pop_k,
                                          tag=f"gen{gen_idx}")

            if cfg.bind_pairs:
                perm = list(range(cfg.pop_size))
            else:
                perm = list(rng.permutation(cfg.pop_size))
            if batch_eval_fn is not None:
                t_e = time.time()
                fits = batch_eval_fn(pop_v, pop_c, pop_k, perm)
                print(f"[gen {gen_idx}] eval {len(fits)} pairs in "
                      f"{time.time()-t_e:.1f}s (parallel)  "
                      f"best={min(fits):+.4f} med={float(np.median(fits)):+.4f}",
                      flush=True)
            else:
                fits = _eval_pairs(
                    pop_v, pop_c, pop_k, perm, fitness_pair_fn,
                    prefix=f"[gen {gen_idx}]",
                )

            f_arr = np.asarray(fits, dtype=np.float64)
            f_arr = np.where(np.isfinite(f_arr), f_arr, np.inf)
            log = TwoPopGenLog(
                gen=gen_idx,
                best_fit=float(f_arr.min()),
                mean_fit=(float(np.mean(f_arr[np.isfinite(f_arr)]))
                          if np.isfinite(f_arr).any() else float("inf")),
                median_fit=float(np.median(f_arr)),
                elapsed_s=time.time() - t0,
                v_attempts=va, v_invalid=vinv,
                c_attempts=ca, c_invalid=cinv,
            )
            history.append(log)
            if on_generation is not None:
                on_generation(log, pop_v, pop_c, pop_k, perm, fits)

        # ---- 4. Build best genome from final pairing ------------------
        best_idx = int(np.argmin(np.where(np.isfinite(fits), fits, np.inf)))
        best_v = pop_v[perm[best_idx]]
        best_c = pop_c[best_idx]
        best_k = pop_k[best_idx]
        from .program import deep_copy_program
        best_genome = Genome(
            prog_v2c=deep_copy_program(best_v),
            prog_c2v=deep_copy_program(best_c),
            log_constants=best_k.copy(),
        )
        return TwoPopResult(
            best_genome=best_genome,
            best_fitness=float(fits[best_idx]),
            pop_v=pop_v, pop_c=pop_c, pop_k=pop_k,
            pair_perm=perm, fits=fits,
            history=history,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _eval_pairs(
    pop_v, pop_c, pop_k, perm, fitness_pair_fn, *, prefix=""
) -> List[float]:
    """Evaluate `len(pop_c)` triples (v[perm[i]], c[i], k[i])."""
    from .program import deep_copy_program
    n = len(pop_c)
    fits: List[float] = []
    for i in range(n):
        g = Genome(
            prog_v2c=deep_copy_program(pop_v[perm[i]]),
            prog_c2v=deep_copy_program(pop_c[i]),
            log_constants=pop_k[i].copy(),
        )
        f = float(fitness_pair_fn(g))
        fits.append(f)
        if prefix:
            print(f"{prefix} pair {i:>3d}/{n}  fit={f:+.4f}", flush=True)
    return fits


def _evolve_constants(
    pop_k: List[np.ndarray],
    fits_k: List[float],
    cfg: "EvolutionConfig",
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Constants pool: tweak top survivors, dedup by quantized hash."""
    n = cfg.pop_size
    order = np.argsort(np.where(np.isfinite(fits_k), fits_k, np.inf))
    new_pop: List[np.ndarray] = []
    seen: set = set()
    if cfg.elitism > 0:
        for i in order[: cfg.elitism]:
            k = pop_k[int(i)].copy()
            fp = _const_fingerprint(k)
            if cfg.dedup and fp in seen:
                continue
            seen.add(fp)
            new_pop.append(k)
    while len(new_pop) < n:
        parent_idx = int(tournament_select_idx(fits_k, cfg.tournament_k, rng))
        k = pop_k[parent_idx].copy()
        # Always tweak to ensure diversity (offspring should differ).
        k = mutate_log_constants(k, rng,
                                 n_tweaks=max(1, int(rng.integers(1, 3))))
        fp = _const_fingerprint(k)
        if cfg.dedup and fp in seen:
            # Tweak harder
            k = mutate_log_constants(k, rng, n_tweaks=3, sigma=0.6)
            fp = _const_fingerprint(k)
            if fp in seen:
                continue
        seen.add(fp)
        new_pop.append(k)
    return new_pop


def tournament_select_idx(
    fits: List[float],
    k: int,
    rng: np.random.Generator,
) -> int:
    """Index-returning tournament select (smaller fitness wins)."""
    n = len(fits)
    if n == 0:
        raise ValueError("empty pop")
    candidates = rng.integers(0, n, size=min(k, n))
    best_i = int(candidates[0])
    best_f = fits[best_i]
    for ci in candidates[1:]:
        cf = fits[int(ci)]
        if (np.isfinite(cf) and (not np.isfinite(best_f) or cf < best_f)):
            best_i = int(ci)
            best_f = cf
    return best_i


def _evolve_triple_offspring(
    *,
    pop_v: List, pop_c: List, pop_k: List[np.ndarray],
    fits: List[float],
    cfg: "EvolutionConfig",
    rng: np.random.Generator,
    rpg: RandomProgramGenerator,
    pool=None,
    n_workers: int = 1,
) -> Tuple[List, List, List[np.ndarray], int, int, int, int]:
    """Build next-generation populations as **bound triples**.

    For each output slot j a single parent triple index ``a`` is picked
    by tournament on ``fits`` (which is per-pair).  With prob
    ``p_crossover`` a second triple ``b`` is picked.  The new
    ``(v', c', k')`` triple is built by:

      * v' = crossover_program(pop_v[a], pop_v[b])  (or copy of pop_v[a])
        followed by mutate_program (V2C instruction set)
      * c' = crossover_program(pop_c[a], pop_c[b])  (or copy of pop_c[a])
        followed by mutate_program (C2V instruction set)
      * k' = crossover_log_constants(pop_k[a], pop_k[b]) (or copy of
        pop_k[a]) followed by mutate_log_constants with prob
        ``p_const_tweak``

    Validation gates entry: both v' AND c' must pass their respective
    validators (the constants vector is bounded by construction).  If
    either fails, the candidate is rejected and a fresh parent draw is
    performed.

    Elitism: top ``cfg.elitism`` triples (by pair fitness) are copied
    untouched into the first slots.

    Returns ``(new_pop_v, new_pop_c, new_pop_k, v_attempts, v_invalid,
    c_attempts, c_invalid)`` — the per-side attempt / invalid counters
    are kept for compatibility with ``TwoPopGenLog``.
    """
    from .crossover import crossover_log_constants
    from .program import deep_copy_program
    from .mutation import mutate_log_constants, mutate_program

    n = cfg.pop_size
    order = np.argsort(np.where(np.isfinite(fits), fits, np.inf))
    rank_of = {int(idx): r for r, idx in enumerate(order)}

    new_v: List = []
    new_c: List = []
    new_k: List[np.ndarray] = []
    # Behavioral fingerprint dedup (per side, mirrors per-side path).
    seen_v: set = {_behav_fingerprint("v2c", p) for p in pop_v}
    seen_c: set = {_behav_fingerprint("c2v", p) for p in pop_c}

    # ----- Elitism: copy top-k triples untouched ----------------------
    if cfg.elitism > 0:
        for i in order[: cfg.elitism]:
            ii = int(i)
            new_v.append(deep_copy_program(pop_v[ii]))
            new_c.append(deep_copy_program(pop_c[ii]))
            new_k.append(pop_k[ii].copy())

    v_attempts = 0
    v_invalid = 0
    c_attempts = 0
    c_invalid = 0
    target = n
    max_attempts = cfg.max_attempts_per_slot * target

    while len(new_v) < target:
        if v_attempts + c_attempts >= max_attempts * 2:
            raise RuntimeError(
                f"triple offspring fill exhausted: "
                f"{len(new_v)}/{target} after "
                f"v_attempts={v_attempts} c_attempts={c_attempts}"
            )
        a = tournament_select_idx(fits, cfg.tournament_k, rng)
        n_mut = _rank_to_n_mutations(rank_of.get(a, 0), n, cfg)
        if rng.random() < cfg.p_crossover:
            b = tournament_select_idx(fits, cfg.tournament_k, rng)
            v_child0 = crossover_program(pop_v[a], pop_v[b], rng,
                                          instr_set=V2C_INSTR)
            c_child0 = crossover_program(pop_c[a], pop_c[b], rng,
                                          instr_set=C2V_INSTR)
            k_child = crossover_log_constants(pop_k[a], pop_k[b], rng)
        else:
            v_child0 = deep_copy_program(pop_v[a])
            c_child0 = deep_copy_program(pop_c[a])
            k_child = pop_k[a].copy()

        v_cand = mutate_program(v_child0, rng, rpg, V2C_INSTR,
                                 n_mutations=max(1, n_mut))
        c_cand = mutate_program(c_child0, rng, rpg, C2V_INSTR,
                                 n_mutations=max(1, n_mut))
        if rng.random() < cfg.p_const_tweak:
            k_child = mutate_log_constants(k_child, rng,
                                            n_tweaks=max(1, int(rng.integers(1, 3))))

        # Independent per-side validation.  Both must pass.
        v_attempts += 1
        c_attempts += 1
        v_seed = int(rng.integers(0, 2**31))
        c_seed = int(rng.integers(0, 2**31))
        v_ok = _validate_one("v2c", v_cand, deg=cfg.validator_deg, seed=v_seed)
        if not v_ok:
            v_invalid += 1
            continue
        c_ok = _validate_one("c2v", c_cand, deg=cfg.validator_deg, seed=c_seed)
        if not c_ok:
            c_invalid += 1
            continue
        # Dedup on both sides (independent: a pair is rejected if EITHER
        # side collides with an existing member).
        if cfg.dedup:
            v_fp = _behav_fingerprint("v2c", v_cand)
            if v_fp in seen_v:
                continue
            c_fp = _behav_fingerprint("c2v", c_cand)
            if c_fp in seen_c:
                continue
            seen_v.add(v_fp)
            seen_c.add(c_fp)

        new_v.append(v_cand)
        new_c.append(c_cand)
        new_k.append(k_child)

    return new_v, new_c, new_k, v_attempts, v_invalid, c_attempts, c_invalid

