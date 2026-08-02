# 0005 — 314 micro tokens map to 12 macro families by hand, in `data/`

**Status:** accepted
**Ticket:** P1-11

## Decision

Twelve macro families, one explicit row per token, curated by hand:

```
Hip Hop                     47 tokens    317 tracks primary   348 in set
R&B, Soul & Funk            28           168                  199
Pop                         38           213                  250
Rock                        52           320                  339
Punk & Metal                14            24                   27
Electronic & Dance          42            58                   71
Jazz                        19            42                   75
Latin                       33            53                   64
Reggae & Caribbean           9             7                   11
Folk, Country & Traditional 22            15                   21
Stage & Screen               6            12                   12
Holiday & Novelty            4            23                   29
```

The table is `data/genre_macro_map.json`. It is **source**, not an artifact:

```
data/genre_macro_map.json  ->  hand-written, committed, nothing regenerates it
artifacts/genres/<v>/      ->  build output, gitignored, rebuilt in a second
```

`pipeline/macro.py` loads and validates it; `pipeline/genres.py` applies it and adds
`genre_macro_primary`, `genre_macro_set`, `genre_macro_count` to the rollups.

## Why hand curation, and why that means a committed home

§8.3 picks hand curation because it is the only approach where a macro label can be
explained by pointing at a row. Clustering the tokens or fuzzy-matching them produces a
map whose colours nobody can defend, and the first time a colour looks wrong there is
nothing to correct — only a threshold to nudge.

That property is also the reason the file cannot live under `artifacts/`. Everything in
`artifacts/` is reproducible by rerunning a CLI, which is exactly why it is gitignored;
this file is an hour of judgement that no rerun brings back, and it would be one
`git clean` from gone. `git check-ignore data/genre_macro_map.json` exits non-zero, and
`tests/test_macro_genres.py` runs that check so it stays true.

Because the table is an **input**, its sha256 is folded into the genre version digest
alongside the CSV's. Two artifacts built from the same export under different taxonomies
must not share an id — recurating changes what every colour means.

## Why twelve, and which of Discogs' fifteen did not survive

Modelled on the Discogs genre/style split, adjusted to what this library actually
contains rather than to what a general catalogue needs:

- **Blues** dropped. Three tokens (`blues`, `soul blues`, `jazz blues`), seven tracks,
  and not once at rank 0 — a family that could never be anyone's primary colour. Routed
  to the lineages that grew out of it. If a blues corpus lands, a new family is a table
  edit, not a re-fit.
- **Classical** dropped. `classical`, `opera` and `orchestral` sit on the same **two**
  Laufey tracks and arrive via the Philharmonia Orchestra credit — artist-scope labels
  bleeding onto a jazz-pop song. Two tracks cannot carry a family; they join Stage &
  Screen.
- **Brass & Military**, **Children's**, **Non-Music** dropped — one or zero tracks each.
- **World** deliberately never created. It is the most tempting family here and the
  worst: a bucket whose definition is "not from here", which would have absorbed
  `indian indie` (17 tracks of an indie band), `k-pop`, `malayalam hip hop` and every
  other regional token, and coloured points by passport while the map beneath them is
  organised by sound. Instead the traditional *forms* (chanson, flamenco, bhajan,
  zydeco) join Folk, Country & Traditional, and everything else routes by sound.
- **Punk & Metal** added, splitting Discogs' Rock. Rock is the largest family even after
  the split, and the boundary is one a reader can check by looking at the token.
- **Holiday & Novelty** added. `christmas` is 23 tracks and names an occasion, not a
  sound; folding it into Pop or Jazz would assert something the source did not say.

## The three rules the table runs on

1. **Route by sound, not origin.** `k-pop` is Pop, `indian indie` is Rock, `malayalam
   hip hop` is Hip Hop. The embedding already places points by sound, so a colour keyed
   on origin fights the position it sits at.
2. **Head noun wins.** The last word names the family, the modifiers name the flavour:
   `jazz rap` is Hip Hop, `funk rock` is Rock, `pop country` is Folk/Country.
3. **Except where the compound is its own tradition, or the head noun is a false
   friend.** `lovers rock` is Reggae. `bossa nova` is Jazz. `pop urbaine` is French rap.
   Every exception carries a note in the file.

Ambiguity is resolved against the tracks, not against taste. `riddim` is the clearest
case: the token is a homograph — a dubstep genre and a reggae rhythm — and its single
track is Shaggy's "It Wasn't Me", labelled reggae, dancehall, lovers rock, ragga. The
co-labels settle it; the note records the evidence rather than the reasoning.

## Multi-macro is the point

`genre_macro_set` is a list because **177 of the 1,252 labelled tracks span more than one
family** (163 span two, 12 span three, one four, one five). Collapsing them to a single
value would throw away something the source actually provided: Latin trap really is Latin
*and* Hip Hop; a Sinatra Christmas single really is Jazz *and* a holiday record.

The gap between the two count columns above is where this shows. Jazz is 42 primary but
75 in-set — over half the jazz in this library arrives as somebody else's second label,
and a single-value schema would have hidden all of it.

**177, where P1-11 estimated 184.** The estimate was not a property of the data. It is a
property of a mapping, and this one is different: moving `indie` alone from Rock to Pop
takes the figure to 215, `trap latino` to Latin takes it to 159. It was measured after
curating rather than curated towards, and `tests/test_macro_genres.py` pins the measured
value with that reasoning attached.

## Primary selection

The macro primary is the family of whichever label wins `genres.primary_sort_key` — the
§8.6 chain (confidence, track scope over artist scope, lowest rank, alphabetical), reused
rather than reimplemented. Not "the family with the most labels": that rule would let a
track read *Rock* while its own `genre_micro_primary` was `rap`, with nothing reporting
the contradiction. Reusing the chain also means tier 2's track-scope and sub-1.00 rows
move micro and macro together instead of one of them.

In Phase 1 every label is artist-scope at 1.00, so rank decides and alphabetical is the
tiebreak that never fires — which is precisely why it has to be there. The maximum here
is a 14-label track; without the last two terms that is a 14-way tie resolved by dict
ordering, and the colour would change between builds.

## No `Other`, ever

A token with no row resolves to the sentinel `unreviewed`, which is **not** a declared
family — so it has no colour, no legend entry, and no way to look like a decision. The
build writes `macro_review_queue.json` into the genre artifact with each unseen token's
`track_count`, `first_count` and three example URIs, prints a loud warning, and carries
on. Carries on deliberately: a new export arriving with three new tokens should still
produce a map. The queue file is written on every build even when empty, because a
missing file cannot be told apart from a build that never checked, and the test suite
fails while it is non-empty.

A silent `Other` is how a taxonomy rots. It grows, nobody can say why anything is in it,
and the one thing hand curation was chosen to buy stops being true without anything
breaking.

## Reversing this

Cheap in both directions. Reassigning a token is a one-line edit and a rebuild; the
rollups are derived and `labels.json` is untouched. Adding or splitting a family is the
same edit plus a palette change in P1-12. Nothing about the embedding moves — that is the
whole point of 0004, and this ticket is the first real test of it: a complete genre
taxonomy landed without minting an embedding version.

The expensive direction is the one not taken — deriving macro families by clustering the
tokens. It would have been reversible as code and irreversible as an explanation: once
colours come from a threshold, there is no row to point at, and "why is this green"
stops having an answer.
