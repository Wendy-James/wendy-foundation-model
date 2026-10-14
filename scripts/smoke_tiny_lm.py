"""CPU-only deterministic toy-sequence fitting, not a language model pretraining claim."""
import json
import platform

import torch

from wendyfm.model import ModelConfig, TinyDecoderLM


def run():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=24, d_model=32, n_heads=4, n_layers=2, max_seq_len=12)
    model = TinyDecoderLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).repeat(2, 1)
    losses = []
    for step in range(66):
        model.train()
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        losses.append(round(float(loss.item()), 8))
        if step == 65:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    result = {"type":"toy_sequence_overfit_smoke", "seed":7, "steps":65,
              "params":sum(p.numel() for p in model.parameters()),
              "initial_loss": losses[0], "final_loss": losses[-1],
              "torch":torch.__version__, "python":platform.python_version(), "cpu_threads":1,
              "config":vars(cfg), "final_below_quarter_initial":losses[-1] < losses[0]/4}
    print(json.dumps(result, indent=2))
    if not result["final_below_quarter_initial"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
