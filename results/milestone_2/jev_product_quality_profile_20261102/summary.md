# Milestone 2 Jev Optimized Evaluation

## Execution

- Selected order: `product_defect -> poor_quality -> dissatisfaction`
- Held-out SemCEB reviews: 800
- Completed decisions / successful Jev calls: 952
- Input / output / total tokens: 303333 / 21076 / 324409
- Estimated execution cost: $0.012739986
- Summed inference latency: 95.992702 seconds
- Measured optimized wall time: 29.972815 seconds
- Final matches: 66
- Completion: `complete`

## Per-predicate flow

| Source | Predicate | Input | Passed | Rejected | Observed selectivity |
|---|---|---:|---:|---:|---:|
| Historical fixed | poor_quality | 800 | 91 | 709 | 0.113750 |
| Historical fixed | product_defect | 91 | 67 | 24 | 0.736264 |
| Historical fixed | dissatisfaction | 67 | 66 | 1 | 0.985075 |
| Fresh optimized | product_defect | 800 | 86 | 714 | 0.107500 |
| Fresh optimized | poor_quality | 86 | 66 | 20 | 0.767442 |
| Fresh optimized | dissatisfaction | 66 | 66 | 0 | 1.000000 |

## Historical fixed-order trace comparison

- Fixed decisions: 958
- Optimized decisions: 952
- Decision reduction: 0.626305%
- Fixed total tokens: 326210
- Optimized total tokens: 324409
- Token reduction: 0.552098%
- Fixed estimated cost: $0.012810000
- Optimized estimated cost: $0.012739986
- Cost reduction: 0.546557%
- Fixed summed inference latency: 110.987545 seconds
- Optimized summed inference latency: 95.992702 seconds
- Summed-latency reduction: 13.510383%
- Wall-clock speedup: not calculated; the historical subset has no comparable measured wall time.

## Prediction error

- Predicted optimized evaluations: 914.0
- Observed optimized evaluations: 952
- Predicted optimized cost: $0.012069965
- Observed optimized cost: $0.012739986
- Absolute cost error: $0.000670021
- Relative cost error: 5.259194%

## Profiling overhead

- Profiling calls: 600
- Profiling tokens: 201927
- Profiling cost: $0.007909734
- Gross execution savings: $0.000070014
- Net savings for one query after profiling: $-0.007839720
- Break-even executions: 113

## Output equivalence

- Exact final match-set equality: `False`
- Fixed-only IDs: 1
- Optimized-only IDs: 1
- Shared decision-pair Boolean agreement: 99.176955%
- Shared-pair threshold crossings: 2

Fixed-only final IDs:

- `sf_ac4e6614d925ff19cc446f120c238cd19ac0349f211c18e6ff71fc6eabdd627e`

Optimized-only final IDs:

- `sf_d8442bce1f82303b5e686090c27c35b7b57963d938956e820c2e480a864f88c1`

Shared-pair threshold crossings:

- `sf_ac4e6614d925ff19cc446f120c238cd19ac0349f211c18e6ff71fc6eabdd627e` / `poor_quality`: 0.51 (pass) to 0.48 (fail)
- `sf_d8442bce1f82303b5e686090c27c35b7b57963d938956e820c2e480a864f88c1` / `dissatisfaction`: 0.49 (fail) to 0.51 (pass)

## Final optimized match IDs

- `sf_01555ac7932a56d777c02b7abbf6c65f78906a3d72f86ea122daeb9416ab9bca`
- `sf_026ce942db9fc8c5df50254b48ce1ef05f938fa8296ec22e5f53303d54bd8a73`
- `sf_05778154d7638eff2244579ac0bb147282fd166077320e27863285938481e6e9`
- `sf_08904b0ddb3d10311b52176941cd75f15edf2f596e7d86f22d14f79af90b7482`
- `sf_0d8ebaef9b4a7bbff113bab2f07c023b2d9f65ba20acbc751f4a132a25dd1128`
- `sf_11fc25c71836c008375a96b0130810f466a47eceacb63784588b896f73e68785`
- `sf_15e76b108f35d83e9703b4e58eed2ae84812084e81f19098f5e8fe3c82ca2da9`
- `sf_18141a6468cbc8e6a222d9f1c13363dfc2ef948e2b59286ac8227c505f4894b0`
- `sf_1896cec27164932a5ad1b96013cecc38fe0191e09953b3a13f771671d6cf309b`
- `sf_196d03d31b6ce8a30d01f0214430723e5efb70b9e90734456a3c71af33db11b1`
- `sf_1af3b76f30686585fc85ff8d5c4c5236741deb94d9cb984111987852e655533d`
- `sf_1cee2025e5d89f0ac02c5934cfd7cb96231005d2e1d767133c2ed366454f551c`
- `sf_1eb684567f8ca79b486dedbcb54eb115c97b30f772f9ec8503f57851dbbcbd0b`
- `sf_2139bb71509f7b0652d10b128488af0bf3e8046c1386483d65bf9b6a26259d85`
- `sf_2444cae4acce3b364cb00cdd9df7489393ce6f25289de57b05d4e0f9ed699ebb`
- `sf_3523947854feed573c073854a10d20eaa88a1086b7fc3a771d4f83397ea176ce`
- `sf_354a585b2ce2a840b5d349347cc19167427ba492a1f35c1c00352c4c928112b0`
- `sf_4021e91bc06fae255f1baf51e5b07a0c9a6d9ad18c1cc0de533b3922d19990dd`
- `sf_419831856cc989e840c32bd734c9754c34441e18c816b5cec0a5d34e4fd84981`
- `sf_471c9d55633ee579daff87281b5e6ef043933d4164afdc90970d00074a622a4f`
- `sf_47c7a12add96118494dc4b49aadfed9049121cda009ac46388764f3f36141f6d`
- `sf_48a9c31cb2143bd699585911e36164ac1e7b072426dac4523b0e0930842d22bf`
- `sf_54fc9a684749459b33446eff669bf156ec5db7e975dcc512e0d6a8e3ecdfe02c`
- `sf_570f5f870fa34f6b7db92712b86249aa26683888cb259326c93b627c7d829ae0`
- `sf_57a7c34bd31c387a8830d1a18b5e2a4fef2a91cac4e3963617d1b2f9927c8722`
- `sf_5bd2fa98d80f00e0dd845c3a2fbe15763630d76adc22677a3d52c5a0d3d91263`
- `sf_5e0039bd4b5553424d91fa61af77e7022f8bb39e35e64b2783129bde4c705c7f`
- `sf_5f2c712f0e638a29868966997a1e71ca9510d877067c3298937a2c6116b8ff09`
- `sf_6668dd73dbde3fcea1a77ef535296a0e20e7f933b1e6743d4b8dde235faf09d7`
- `sf_66a9a2c194afad648b0fe75b923d67d8caed5e1af264fa5f36e072eb69c28e0d`
- `sf_68a29d1d874f00fcf2e5cb25ad8c2224534b3632477481d0e43a24dd02ed9ccd`
- `sf_6ab4609357264cda79eaf823982d7cd91f75627d56cceda93e7e93b31fe80289`
- `sf_6b7c41557a218e980db2fc78d9f1a180a0be9553243b77fefffcab4f722ba3a2`
- `sf_6c590ee04681f0fc483ef3e40688250d39422169672fc3189d5fbee0cf0100fd`
- `sf_6c80965d58b4d57e40d60f8334b25573273ce396e394d741f11a219da1e6c3d3`
- `sf_816aa4c93b528260caa3d5e2f5a23a546e682e10ec6beafbe9eae62ac5a97a45`
- `sf_817ac0cbf3b80338288f12eeadc89be48bfc10c5f46a5c198cfa3d4e12d16baa`
- `sf_8528c492c7d40d8fbc4d918d2a8bd61a54f9fc400923e134428c7a6ae5a6f7bc`
- `sf_85f6699001f4f794a6cdfd5cf1e7bcd6600300b984e99ed2843c7556e12262e4`
- `sf_8702efa75f7372bbd87a13dd204359d2fa92f4c8a3c65e73ca094c824df38dff`
- `sf_8b31ffaf4daea2d5a914f489da664e0b35335429b10da60b00a9f9b18a292547`
- `sf_8b67fe5715ec30143692827963bd1bc8a010fe0df1edeced2d7eb3e5e78569ac`
- `sf_9403f82c9c0df46bdbf1523df355d6f415376fdc6c1f419f54de597a24e37ad7`
- `sf_94539ad4f23b29b56e0575e5b58227e8fbde9289a239a0c14448c35d8f8fecfc`
- `sf_9f0793615f9ae6186b774b4a03511d5b8d07ffe52b1e3dc205aa9c3e13c71174`
- `sf_a2cd9c7bdc6cbd7004d90909f0e1a3c5acb3adf384f30713b45b6a5ede50cfdd`
- `sf_a3048446a7c6e5303d54c03a34d89fb7aff7df61c921916a0967d937c2f81fd7`
- `sf_a3dc11f514f70a9169b28b8a17011800a4bf2c99efed6ce91c94a7bf6e89699c`
- `sf_b1cea71a5c19b77bebf10e6c3cd53e0afa1fd063a854d702a4e8616b361cc2ef`
- `sf_b7e6b71d8f979ee563a866ebb48b9ee35c32224b8da483212ad475462e279bd1`
- `sf_b8aedc73e7a173c092a8074a74118213e818ca67df819e891177d485e442ad1f`
- `sf_b9ae89f235cad633424a3c72f6fb31e4ca8228bafae2846aced2cf02c6cf8175`
- `sf_c703796c42f3f0fd0f1f92a5f250ae8244d7c73b0e8c521cbe5336a63cdff037`
- `sf_c99654dd8c228e93a8aba7fa79588d228fdf222984f29da95e26c423e54f2396`
- `sf_cb9ec7504e1b5575e6c2b81ff8a0f28e4fd586650edb025316a411c4cdffbb0b`
- `sf_d456cc28f8ee7e5a24220244cd108aae006931dd32739f5b319096e52568c66b`
- `sf_d8442bce1f82303b5e686090c27c35b7b57963d938956e820c2e480a864f88c1`
- `sf_ddb687a773547532fef66a622415dade98f9806cc97ffbd9d4cab5b0faa20cec`
- `sf_dddf59be4bc7bff32ec3548d14f277be6ba3d75a336093d25384e4741839e26f`
- `sf_e173e7c4c238595ddee311d78cd13e2cb649907fb9660ee8e0864359eb3f4fc2`
- `sf_e4635d377e006207b681d6dda1a32cd6ec281ff2c4f40da677da804db1259977`
- `sf_eb04287995d7721d657b19ffa70606afa9170949141edc94a8236ff8bffc190c`
- `sf_eb972e9dbf8db2f9812756d416af9e163ca255b5777774347cd495b5a252d14e`
- `sf_efa7d6e163020b1fda93ff96d220853df07f8fb6e600e8362ea5592f4b1d2542`
- `sf_f8051f7bed754c74d7e6f0bec3c0538edb6cffcfbcb13613f3e6a1a7ba337426`
- `sf_fdf86a1d1b4e75761361bb3a6febbb2272238968e5c598cefd0df8ca4b0ef181`

## Authorization accounting

- Absolute authorized attempt limit: 2640
- Preserved pre-run transport failures: 12
- Successful run attempts: 952
- Total counted attempts: 964
- Remaining authorization: 1676

The failed transport attempt produced no model decisions, tokens, or cost and is preserved separately. No new fixed-order run was performed.
