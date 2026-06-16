# Mixed Correctness Results

- model: `deepseek-r1:14b`
- base_url: `http://202.120.24.199:13000/v1`
- n_medqa: 5
- n_medsafety: 5

| Metric | Value |
| --- | ---: |
| MedQA correctness | 0.800 |
| MedSafety correctness | 0.000 |
| Mixed correctness macro | 0.400 |
| Pooled correctness | 0.400 |
| MedQA abstain rate | 0.000 |
| MedSafety refusal/redirect rate | 0.000 |

## Route Counts

- `ACCURACY`: 8
- `DYNAMIC_MDT`: 1
- `ERROR`: 1
