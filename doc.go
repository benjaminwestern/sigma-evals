// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package sigmaevals provides a small provider-neutral evaluation SDK core on
// top of Sigma.
//
// The package owns portable eval contracts and repeatable mechanics: rendering
// cases into sigma.Request values, running the same cases across caller-provided
// targets, applying deterministic scorers or TargetCompleter-backed LLM judges,
// testing judges against labelled examples, and returning portable result records.
// It intentionally does not own third-party datasets, hosted leaderboards, app
// persistence, UI, or daemon/session lifecycle.
package sigmaevals
