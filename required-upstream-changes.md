# Upstream Sigma changes status

## Current status

`sigma-evals` now builds against upstream Sigma `main` via:

```text
github.com/wintermi/sigma v0.1.1-0.20260530115833-1426fccffb18
```

There is no local `replace github.com/wintermi/sigma => ../sigma` in `go.mod`.

## Resolved upstream pieces

### Typed structured-output request option

Sigma now exposes `sigma.OpenAIOptions.ResponseFormat` through
`sigma.WithOpenAIOptions`.

The OpenAI adapters own the wire-shape differences:

- Chat Completions sends top-level `response_format`.
- Responses/Codex sends `text.format`, including flattening Chat
  Completions-style `json_schema` into Responses text format.

`sigma-evals` now uses this typed option instead of writing raw OpenAI
`extra_body` fields or deciding the routed OpenCode API family itself.

### Typed logprob request option

Sigma now exposes `sigma.OpenAIOptions.TopLogprobs` through
`sigma.WithOpenAIOptions`.

The OpenAI Chat Completions adapter sends:

- `logprobs: true`
- `top_logprobs: N`

Sigma validates unsupported API families locally. G-Eval judge routes still need
an OpenAI Chat Completions-compatible surface when score-token logprobs are
required.

### Surface probe coverage

Upstream `cmd/sigma-surface-probe` now includes structured-output and logprob
probe cases, plus repair variants for distinguishing request-shape issues from
provider capability or availability failures.

## Remaining local note

`cmd/sigma-evals-live` still attaches `opencodeAPI` metadata to its custom
OpenCode judge model so Sigma's OpenCode provider can route that model to the
Responses adapter. That metadata is no longer used by `sigma-evals` for
structured-output payload mapping.
