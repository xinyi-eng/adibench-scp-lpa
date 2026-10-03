# ADIBench

A multi-granularity benchmark for evaluating **few-shot learning** methods on
**long-tail Arabic dialect identification** (ADI).

## Datasets

| Config        | Classes | Size    | Granularity    | Domain    |
|---------------|---------|---------| |------------|
| `nadi_18`     | 18      | 440,052 | Country       | Twitter   |
| `nadi_5`      | 5       | 440,052 | Coarse group  | Twitter   |
| `amgadhasan_5`| 5       | 147,725 | City          | Twitter   |

## Installation

```bash
pip install datasets transformers torch
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("adibench/nadi_18", split="train")
print(ds[0])  # {'text': '...', 'label': 5}
```

## Evaluation

We provide a few-shot evaluation harness compatible with `huggingface/evaluate`:

```python
import evaluate
metric = evaluate.load("adibench/accuracy")

# predictions: (N, n_classes) array of probabilities
# references:  (N,) array of int labels
results = metric.compute(predictions=probs, references=labels)
print(results)
# {'accuracy': 0.42, 'ece': 0.18, 'brier': 0.62, 'n_samples': 1500}
```

## Leaderboard

Coming soon. Submit your results by opening a Pull Request on the
`leaderboard/` directory with a JSON file of the form:

```json
{
  "method": "MyMethod",
  "date": "2026-09-15",
  "results": {
    "nadi_18_1shot": 0.42,
    "nadi_18_5shot": 0.55,
    "nadi_5_1shot": 0.51,
    "nadi_5_5shot": 0.69,
    "amgadhasan_5_1shot": 0.66,
    "amgadhasan_5_5shot": 0.78
  }
}
```

## License

Research use only. The underlying NADI 2024 and amgadhasan datasets have
their own licences — please consult the original providers.

## Citation

```
@inproceedings{adibench2026,
  title={ADIBench: A Multi-Granularity Benchmark for Few-Shot Arabic Dialect Identification},
  author={Anonymous},
  booktitle={Anonymous submission},
  year={2026},
}
```