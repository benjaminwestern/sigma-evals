// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestToolCallScorerRejectsWrongArguments(t *testing.T) {
	t.Parallel()

	score, err := sigmaevals.ToolCallScorer{}.Score(context.Background(), sigmaevals.ScoreInput{
		Case: sigmaevals.Case{ID: "tool", Expected: sigmaevals.Expected{ToolCalls: []sigmaevals.ExpectedToolCall{
			{Name: "read_file", Arguments: map[string]any{"path": "config/app.toml"}},
		}}},
		Message: sigma.AssistantMessage{Content: []sigma.ContentBlock{
			sigma.ToolCallBlock("call_1", "read_file", map[string]any{"path": "config/other.toml"}),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if score.Passed || score.Score != 0 {
		t.Fatalf("score = %+v, want failing tool call score", score)
	}
}

func TestRunnerCanScoreToolCallOnlyOutput(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{{Content: []sigma.ContentBlock{
		sigma.ToolCallBlock("call_1", "read_file", map[string]any{"path": "config/app.toml"}),
	}}}}
	result, err := sigmaevals.NewRunner(client).Run(context.Background(), sigmaevals.RunSpec{
		Suite: sigmaevals.Suite{Name: "tools", Cases: []sigmaevals.Case{
			{
				ID:    "read-config",
				Input: "Read config/app.toml.",
				Expected: sigmaevals.Expected{ToolCalls: []sigmaevals.ExpectedToolCall{
					{Name: "read_file", Arguments: map[string]any{"path": "config/app.toml"}},
				}},
			},
		}},
		Models: []sigma.Model{{ID: "model", Provider: "test"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Passed != 1 || result.Summary.Errors != 0 {
		t.Fatalf("summary = %+v, want one passing tool-call eval", result.Summary)
	}
}
