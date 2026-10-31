# TODO: DemPar directed cut rule misses routes against the arrows

Found 2026-10-31 while checking the cut rules in the causal-graph tool
(`~/Toy-Causal-Synthetic-Data-Generator`, fixed there in index.html and headless/csdg.py).
Not yet fixed in this repo.

## The problem

`DPFairness.select_biased_edges` (src/causal_fairness_sdg/fairness/dp.py) calls
`biased_edges_from_paths` with no blocking, so it cuts the last edge of every *directed*
path S -> ... -> Y. Its docstring says DemPar is DECAF Cor. 3, i.e. CF with an empty
admissible set, and the code does not do that. Routes that run against the arrows are
never cut:

| DAG | dp.py cuts | CF rule with R = {} (DECAF Cor. 3) |
|---|---|---|
| Y -> S | nothing | Y -> S |
| Z -> S, Z -> Y (common cause) | nothing | Z -> Y |
| Y -> X -> S | nothing | Y -> X |
| S -> C <- Y (shared child) | nothing | S -> C, Y -> C |

So on such graphs DemPar can cut less than CF, although DemPar should be the strictest rule.

## Does it touch our real-data results?

On the 11 hand-drawn dataset graphs (as loaded in the tool's dataset presets), only one
DemPar cut set changes:

- **cdc_diabetes:** Age (admissible) -> Education, Age -> Income (both protected) and
  Age -> Diabetes_binary. dp.py leaves Age -> Diabetes_binary; DECAF Cor. 3 cuts it.

The other 10 hand-drawn graphs give the same DemPar cuts under both rules. FTU and CF are
unaffected. The MST (undirected) DemPar rule in dp.py `allow_edge` is unaffected.
Not checked: the FGES-discovered graphs, the graph-variant batches (sweep notes §23),
and graphs a generator learns itself.

## Fix when we get to it

In dp.py, replace the path rule with the CF Markov-boundary test using an empty admissible
set (same loop as `CFFairness.select_biased_edges`, `admissible = set()`). Then rerun the
DemPar arms for cdc_diabetes on directed-cut generators, and check the FGES and variant
graphs for any other changed cut sets before deciding what else to rerun.
