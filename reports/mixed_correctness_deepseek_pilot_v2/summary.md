# Mixed Correctness Results

- model: `deepseek-r1:14b`
- base_url: `http://202.120.24.199:13000/v1`
- n_medqa: 5
- n_medsafety: 5

| Metric | Value |
| --- | ---: |
| MedQA correctness | 0.600 |
| MedSafety correctness | 0.800 |
| Mixed correctness macro | 0.700 |
| Pooled correctness | 0.700 |
| MedQA abstain rate | 0.000 |
| MedSafety refusal/redirect rate | 0.800 |

## Route Counts

- `ACCURACY`: 2
- `DYNAMIC_MDT`: 5
- `SAFETY_REVIEW_LOW`: 3
