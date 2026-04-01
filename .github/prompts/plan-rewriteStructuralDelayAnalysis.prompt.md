# Plan: Rewrite Structural Delay Analysis for 2-D Polar Product Code

**TL;DR**
Redefine structural delay accounting for continuous source information arrival at rate $R_s$. Calculate the wait time for continuous source bits at the encoder and the start-of-decoding time at the receiver based on the transmission rate $R_t$, $N_h, N_v$, modulation, and antennas, rather than incorrectly claiming 0 delay. Update response and manuscript to reflect realistic encoding block accumulation and partial interleaving delays, while improving writing style to minimize AI footprint.

**Steps**
1. **Redefine Variables**: Standardize $N_h$ and $N_v$ as horizontal/vertical code lengths, $R_s$ as source information rate (bits/s), $R_t$ as transmission channel symbol rate (symbols/s), $m$ as modulation order (bits/symbol), and $N_{tx}$ as the number of transmit antennas in both `response.tex` and `main.tex`.
2. **Derive Delay Formula (Encoder)**: Formulate the structural delay of the proposed IPD scheme with partial S-T interleaving, considering parallel and continuous source information arrival (K_s streams). Also consider modulation order and antenna count. Refer to the paper in the reference folder for definition of structural delay. 
4. **Rewrite `response.tex`**:
   - Locate parts claiming $L_{struct} = 0$.
   - Remove highly structured bullet points and rewrite as continuous, formal, extended paragraphs (reducing AI footprint).
   - Insert the derived structural delay equations $f(N_h, N_v, R_s, R_t, m, N_{tx})$.
5. **Update Manuscript (`main.tex`)**:
   - Correct the overly optimistic "0 latency" claims in Definition 1 and Remark 1.
   - Integrate the refined mathematical latency derivations into the performance analysis section.
6. **Compile and Verify**: Run `pdflatex` on both files to generate final valid PDFs.

**Relevant files**
- `response/response.tex` — Modify delay analysis text to be paragraph-heavy and mathematically sound.
- `manuscript/main.tex` — Update temporal decoding definitions, theorem texts, and delay formulas.

**Verification**
1. Run `pdflatex main.tex` (twice for references) and check for zero compilation errors.
2. Run `pdflatex response.tex` and ensure LaTeX correctness.
3. Manually search the updated `.tex` files to guarantee words like "0 structural delay" are removed, and bullet points in the modified response sections are flattened into smooth prose.

**Decisions**
- Assumed continuous source arrival at $R_s$ bits/second aligns with the classic structural delay definition for block codes vs LDPC, requiring wait times at the encoder.
- Decoder structural delay will respect the physical transmission time $T_{sym} = 1/R_t$ per symbol.
