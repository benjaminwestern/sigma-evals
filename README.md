# sigma-evals

`sigma-evals` is a small Go SDK for provider-neutral LLM evaluation on top of
[Sigma](https://github.com/wintermi/sigma). It gives applications portable eval
contracts, runners, scorers, judges, and result records without forcing a hosted
service, dataset format, UI, or persistence layer.

Use it when you want the same suite to run against several model providers,
agent runtimes, or saved outputs, while keeping the scoring and result shape
stable.

## Status

Public preview. The module builds from a fresh clone and pins Sigma by the
pseudo-version in `go.mod`; there is no local `replace` dependency.

## Quick start

Run the bundled smoke examples without provider credentials:

```bash
git clone https://github.com/benjaminwestern/sigma-evals.git
cd sigma-evals
mise run ci
```

Use it from another Go module:

```bash
go get github.com/benjaminwestern/sigma-evals
```

Run the example CLI directly:

```bash
go run ./cmd/sigma-evals smoke-examples \
  --examples examples \
  --out runs/examples-smoke.json
```

Run one suite against a real Sigma target:

```bash
go run ./cmd/sigma-evals run-suite \
  --suite examples/generic/answer-aliases.json \
  --target fireworks=accounts/fireworks/routers/kimi-k2p6-turbo \
  --out runs/answer-aliases.json
```

Judge an existing output through the same target seam:

```bash
go run ./cmd/sigma-evals judge-output \
  --judge openai=gpt-4o \
  --target-output "Bonjour" \
  --ground-truth "Bonjour" \
  --rubric "Grade exact translation correctness."
```

## What it provides

- A portable `Suite` / `Case` / `Expected` JSON model for text, chat, multiple-choice, JSON, and tool-call evals.
- SDK runners for direct Sigma models, caller-provided targets, raw fanout, and scoring existing outputs without regenerating completions.
- Deterministic scorers for exact, normalised, contains, regex, JSON structure, token F1, multiple choice, pass@k, and expected tool calls.
- LLM-as-judge helpers for strict JSON judges, pairwise judging, G-Eval score-token weighting, and weighted rubric scoring.
- Judge-alignment evaluation with regression, classification, calibration, and tolerance metrics against labelled examples.
- Small CLI consumers under `cmd/` plus working JSON examples under `examples/` for local smoke tests and integration reference.

## How it fits together

The main seam is `TargetCompleter`. Apps can plug in direct Sigma calls,
agentic sessions, local models, CI jobs, or saved runtime traces while
`sigma-evals` handles rendering, repeats, concurrency, scoring, aggregation, and
result records.

```go
type TargetCompleter interface {
    CompleteTarget(context.Context, sigmaevals.TargetRequest) (sigmaevals.TargetResult, error)
}
```

For direct Sigma use, adapt a Sigma client:

```go
runner := sigmaevals.NewTargetRunner(
    sigmaevals.NewSigmaTargetCompleter(sigma.NewClient()),
)

result, err := runner.Run(ctx, sigmaevals.TargetRunSpec{
    Suite: suite,
    Targets: []sigmaevals.Target{{
        Provider: sigma.ProviderFireworks,
        ModelID:  "accounts/fireworks/routers/kimi-k2p6-turbo",
    }},
    Scorers: []sigmaevals.Scorer{sigmaevals.AutoScorer{}},
})
```

LLM judges use the same seam, so agent runtimes do not need to pretend to be a
`sigma.Client`:

```go
evaluator := sigmaevals.NewTargetEvaluator(myTargetCompleter)

judge, err := evaluator.Judge(ctx, sigmaevals.JudgeInput{
    TargetOutput:  generatedAnswer,
    GroundTruth:   "Bonjour",
    Rubric:        "Grade exact translation correctness.",
    Judge:         sigmaevals.Target{Provider: "agent-runtime", ModelID: "judge"},
    Mode:          sigmaevals.ModeGEval,
    PassThreshold: 4.0, // optional; defaults to 3 on the 1-5 G-Eval scale
})
```

## Example suites

Working JSON suites live in [`examples`](examples). They are intentionally small
showcases, not serious model-quality benchmarks.

- [`examples/generic`](examples/generic) demonstrates closed-answer alias matching, negative-answer rejection, and needle retrieval.
- [`examples/chat`](examples/chat) demonstrates single-turn JSON extraction and multi-turn context recall.
- [`examples/choice`](examples/choice) demonstrates multiple-choice selected-label scoring.
- [`examples/tools`](examples/tools) demonstrates expected tool-call evaluation by tool name and JSON arguments.

## CLI consumers

The CLIs under [`cmd`](cmd) are reference consumers of the SDK interfaces, not a
hosted product surface.

- [`cmd/sigma-evals`](cmd/sigma-evals) runs local smoke examples, real suites against Sigma targets, and one-off LLM judge checks for existing outputs.
- [`cmd/sigma-evals-live`](cmd/sigma-evals-live) is an optional live harness for provider-backed needle and tool-calling checks. It requires `FIREWORKS_API_KEY` and `OPENCODE_API_KEY`.

## Scoring methods

The core package covers:

- deterministic single-output scoring
- multiple-choice and JSON/schema-style output scoring
- single-turn and multi-turn chat evaluation
- tool-call and full-trace-aware scoring
- reference-answer judging
- strict JSON LLM-as-judge
- pairwise LLM judging with swapped-order consistency checks
- single-score G-Eval and multi-metric rubric G-Eval
- judge-alignment evaluation
- pass@k aggregation for sampled or code-style tasks
- scoring existing outputs without rerunning target models

## Judge-alignment example

```go
alignment, err := sigmaevals.NewTargetEvaluator(
    sigmaevals.NewSigmaTargetCompleter(client),
).EvaluateJudges(ctx, sigmaevals.JudgeAlignmentSpec{
    Name:         "judge-alignment-smoke",
    JudgeTargets: []sigmaevals.Target{sigmaevals.TargetFromModel(judgeModel)},
    Tolerance:    0.5,
    Cases: []sigmaevals.JudgeAlignmentCase{
        {
            ID:             "correct-answer",
            Input:          "Translate hello to French.",
            TargetOutput:   "Bonjour",
            GroundTruth:    "Bonjour",
            Rubric:         "Grade exact correctness.",
            ExpectedScore:  1,
            ExpectedPassed: true,
        },
    },
})
```

`EvaluateJudges` reports MAE, MSE, RMSE, Pearson correlation, Spearman
correlation, accuracy, precision, recall, F1, balanced accuracy, Cohen's kappa,
Brier score, and tolerance accuracy.

## Verification

```bash
mise run go:fmt
mise run go:test
mise run go:vet
mise run examples:smoke
mise run ci
```
