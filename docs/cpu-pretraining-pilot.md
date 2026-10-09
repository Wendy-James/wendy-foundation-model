# Bounded CPU pretraining pilot — 2026-10-09

## Question and setup

Can the existing WendyFM tokenizer, Transformer, sampler, training loop, and
checkpoint code execute a reproducible tiny pretraining run from text through
held-out evaluation? The baseline is the seeded, randomly initialized model.
The only change is four AdamW optimizer updates on sampled training windows.
This is a pipeline smoke experiment, not a language-model quality benchmark.

The [frozen configuration](../configs/cpu_pilot.json) fixes a 272-ID BPE
vocabulary, one 16-wide Transformer layer, sequence length 8, batch size 2,
four optimizer steps, one CPU thread, and seed 20261009. The BPE tokenizer is
trained only on the eight synthetic [training documents](../data/fixtures/cpu_pilot_train.txt).
Three different [validation documents](../data/fixtures/cpu_pilot_validation.txt)
are encoded with that tokenizer. Each line is one document and receives an
explicit end-of-document token. Validation scores every full length-8 window
in the held-out token stream without updating weights. Windows overlap, so
this loss is a deterministic pilot metric rather than an independent-token
estimate. The tokenizer never trains on validation text.

## Reproduce

From the repository root:

```sh
uv sync --group dev
uv run python scripts/pretrain_cpu.py --config configs/cpu_pilot.json --output-dir outputs/cpu_pilot
uv run pytest -q
```

To exercise checkpoint continuation, run once with `--end-step 2` in a fresh
output directory and then again with `--resume` in that directory. The saved
checkpoint includes model and optimizer states, completed global step, Python
and PyTorch RNG states, sampler state, and configuration fingerprints. The
regression test compares this continuation with an uninterrupted four-step
run using exact tensor equality. Checkpoints stay under ignored `outputs/`.

## Measured result

The committed [experiment manifest](../experiments/cpu_pilot_20261009.json)
is copied from one standalone CLI run, with hashes of the frozen config,
fixtures, and serialized tokenizer. Environment: macOS 26.0 arm64, Python
3.13.13, PyTorch 2.14.1, one CPU thread. Its observations were:

| Metric | Observed value |
| --- | ---: |
| Held-out validation loss before updates | 5.609570 |
| Held-out validation loss after updates | 5.579744 |
| First sampled training batch loss | 5.619709 |
| Last sampled training batch loss | 5.554146 |
| Optimizer steps | 4 |
| Training throughput | 2,366.05 target tokens/sec |
| Training wall time | 0.0270 s |
| Total runner wall time | 0.5808 s |
| Checkpoint continuation | exact match in pilot regression test |

Training throughput counts 32 target tokens (four steps × batch 2 × sequence
8) divided by the training-loop wall time, which includes batch sampling,
forward and backward passes, and AdamW updates. The total runner time also
includes tokenizer training, validation, and checkpoint writing.
Timing varies between machines and runs. Train-batch losses come from different
sampled windows, so their difference is not a fixed-batch improvement measure.

The held-out loss declined slightly in this one tiny run. That shows the
pipeline can execute and measure validation; it does not establish useful
generalization. The project is ready to design a GPU experiment, but its
current training loop explicitly rejects GPU models, so a GPU run needs a
separate implementation and validation step.
