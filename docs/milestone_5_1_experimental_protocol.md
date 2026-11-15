# Milestone 5.1 preregistered experimental protocol

## Hypotheses and comparisons

The experiment asks whether multi-decision batching or selective-first hybrid execution can
reduce token-priced cost or latency without unacceptable decision changes. It compares fixed
separate, existing adaptive, full conjunctive fusion, full batching, fixed hybrid, cost-only
hybrid, quality-constrained hybrid, and profiling-aware hybrid. A valid result may retain
separate execution.

Primary economic comparisons are paired on common records. Request count, tokens, dollars,
summed inference latency, and wall time remain separate. Quality comparisons cover each named
predicate, physical group, and complete query. Enriched diagnostics are not population-rate
estimates.

## Frozen data

Each domain contains 590 new records: 50 development, 40 pilot, 100 selection, and 400 final
evaluation. Exact and near-duplicate overlap with prepared Milestones 1–5 text was excluded.

| Dataset | Dataset fingerprint | Pilot fingerprint | Evaluation fingerprint |
|---|---|---|---|
| Amazon All Beauty | `988170c284a8f32f38e171f8bcf9e4aeb870a37149c09ae5a7c7db4ba43a9a51` | `3aba9159353e33cdaffb72720ea240b8c6f9eb8769f65904f294979717cdf841` | `637cfadffb2f19afec775a97afef6411f8d885fed10995718fe303cf78bf78bd` |
| Banking77 | `e138176cf7da4e7f54ec25b6a4ec50bf8e74c1cc55360c700083d1a39475cf9e` | `f2f80a41c6e7dd36f01b0149aa04e1216a13559c6199b716022b6d53b93b8920` | `0c8139bb3c39019ed27e600ae5b0cdbd24bc60c52541fe49d54598570c63160e` |
| UCI SMS Spam | `0aa8516d1d05e68646c7743dedccd97e7ab77654150bc5ec6af5bc7ca7a25463` | `233fd4a49ab5ce869386d556af6b11632dedb73f87152cabb11bb8e70e74fa9e` | `afe8465451332a9fac2b06ed54df5fe83ab37ce66e2950d220ccc41d3c178617` |

Nine workloads cover 2/3/4 predicates, rare and common events, correlated and weakly related
predicates, similar selectivity, and different prompt lengths. Their definitions and unchanged
instructions are in `configs/milestone_5_1/workloads.yaml`.

## Quality rule

The historical gate is not relaxed: at least 40 records, at least 3 positive reference cases,
agreement at least 0.98, recall at least 0.95, false-negative rate at most 0.05, and
changed-decision rate at most 0.02, with real complete-plan evidence. A new evidence taxonomy
explains why a candidate is ineligible without retroactively changing this gate.

An optional human-label workflow uses true/false/uncertain labels per underlying predicate,
annotator provenance, written guidelines, and adjudication. Separate Jev labels are not human
truth. Multiple annotators are preferred; no annotations are fabricated.

## Stage B pilot

For each dataset, thirty representative pilot records receive four separate references, nested
2/3/4-question batches, and one full conjunctive comparator. Ten development records selected by
a frozen lexical ranking receive two separate references and a two-question batch as an
enriched diagnostic. This produces:

- 360 representative separate requests
- 270 representative batch requests
- 90 representative conjunction requests
- 90 enriched diagnostic requests
- 810 logical/physical requests total
- 2,430 maximum HTTP attempts with two retries

Model `jev-1.14.0` is pinned, cache is disabled, and concurrency is four. Expected usage is
308,763 input and 33,865 output tokens, costing $0.012968030 at the stored input-token price.
The conservative no-retry proxy is 463,144 input tokens and $0.019452045; the retry-inclusive
proxy is $0.058356134. Only request ceilings are hard.

The exact authorization phrase is:

`AUTHORIZE M5.1 STAGE B SMALL LIVE BATCHING PILOT`

Stage C may use only pilot and selection evidence to freeze plans. Stage D final evaluation and
Stage E replication each require new authorization. Final evaluation records cannot be opened
for plan selection.

## Fingerprints

- Protocol: `4b9dedcd2f889cdfcfdaf9852e1196e94434a0096a63b040781a4365a01ede35`
- Workloads: `548852e317048aece4e9139113c62ea99ded1eae2b2960e3078b132d16e330f9`
- Pilot schedule: `2aa173452d1a55d7fc7e97f420ccbd6b38b2b6d5d016bb211a6cbb6746f71680`
- Pilot budget: `6f88e9b4221561c55e0010d468b14b8a36cbfc451e06aec92cb58e3075241be8`

Any change requires a versioned amendment and new authorization manifest.
