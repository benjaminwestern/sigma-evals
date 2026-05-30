# Example eval suites

These examples are small, working suites for demonstrating `sigma-evals` surfaces.
They are not good evaluations of real model quality and are not intended to be canonical suites or leaderboards. They are also used by `mise run examples:smoke` to exercise the SDK and CLI consumer without network calls.

## Generic examples

The suites under [`generic`](generic) demonstrate closed-answer alias matching,
negative-answer rejection, and needle retrieval from distracting context.

## Chat examples

The suites under [`chat`](chat) demonstrate single-turn and multi-turn chat evals.
They use deterministic JSON and answer scorers.

## Multiple-choice examples

The suites under [`choice`](choice) demonstrate selected-option scoring from a
choice label or answer text.

## Tool-call examples

The suites under [`tools`](tools) demonstrate checking assistant tool-call
requests structurally by tool name and JSON arguments.
