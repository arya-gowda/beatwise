# 0002 — Key is not in the embedding

**Status:** accepted
**Ticket:** P1-02

## Decision

Musical **Key is dropped**. The embedding uses the ten features in
`pipeline/features.py:FEATURES` and nothing else. Recorded in every manifest as
`key_encoding: "dropped"`.

## Why this needed deciding

Three conflicting definitions existed in the repo:

- `map_visualization_3d.py:24` builds a `final_features` frame containing one-hot Key,
  then `:29` scales `df_clean[umap_features]` alone and `:41` fits on that. **Key is
  silently dropped.** The frame is dead code. The concept doc asserted Key was included;
  it never was.
- The same script's evident *intent* was one-hot Key.
- `ml/data.py` encodes Key as two circular dimensions on the circle of fifths, which is
  what the metric-learning work was validated against.

## Reasoning

**The measured evidence says Key carries almost no genre signal.** In the unregularised
diagonal metric-learning model, Key received the lowest weight of all eleven features
(0.161, against 3.5–6.3 for the top six). At the validated operating point it sits at
0.836, still below average. Whatever Key contributes, it is not what makes tracks feel
alike.

**Dropping it makes Phase 1 verification exact.** The prototype already drops Key, so the
artifact reproduces the prototype's map identically rather than approximately. The
concept doc's verification wording — "reproduces the cluster structure the existing
script produces" — can be read as visual identity, which is a far stronger check than a
topological one.

**Adding it would change the map for no measured benefit.** That is the whole argument.

## Reversing this

Change `FEATURES` and `KEY_ENCODING`, then rebuild. The manifest records both, so
artifacts built under different encodings are distinguishable and a mismatched reducer
raises in `pipeline/project.py` rather than silently producing wrong coordinates.

Note that circle-of-fifths encoding (two dims, shared weight) is musically correct and
already implemented in `ml/data.py` if this is ever revisited — one-hot is the option
worth *not* returning to, since it makes musically adjacent keys equidistant.
