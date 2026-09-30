# WendyFM Agent Instructions

## Mission

Build WendyFM into a rigorous, interview-ready foundation-model engineering project.

The project should demonstrate real understanding of:

- tokenization
- decoder-only Transformer architecture
- pretraining
- optimization
- evaluation
- GPU performance
- distributed training
- scaling experiments

The primary audience is foundation-model internship interviewers and graduate admissions reviewers.

## Engineering principles

1. Prefer simple, correct implementations before optimization.
2. Every important component should have tests.
3. Do not hide core model logic behind high-level model libraries.
4. Controlled experiments must clearly state:
   - baseline
   - change
   - metric
   - result
   - interpretation
5. Avoid unnecessary dependencies.
6. Never commit secrets, tokens, private datasets, model checkpoints, or company data.
7. Do not fabricate experimental results.
8. Do not claim benchmarks that were not actually run.
9. Keep public code reproducible.
10. Prioritize technical depth over demo UI.

## Git workflow

Use meaningful conventional commits such as:

- feat(tokenizer):
- feat(model):
- feat(training):
- test:
- perf:
- exp:
- docs:
- fix:

Do not create meaningless commits only to increase contribution counts.

Before committing:

1. inspect git diff
2. run relevant tests
3. verify no secrets or large artifacts are staged
4. use a descriptive commit message

Do not push company code or internal data into this repository.

## Project scope

Initial scope:

1. tokenizer
2. Transformer implementation
3. training loop
4. small-scale pretraining
5. profiling
6. systems optimization
7. scaling experiments

Post-training and RL may be added later, but they are not part of the first milestone.

## Learning requirement

Codex may assist implementation, debugging, testing, and refactoring.

However, implementations should remain understandable enough that the repository owner can explain:

- tensor shapes
- forward computation
- backward implications
- optimization decisions
- performance bottlenecks
- experimental conclusions

