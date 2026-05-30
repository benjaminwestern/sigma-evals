// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
)

func TestAnswerScorerNegativeAnswerWins(t *testing.T) {
	t.Parallel()

	score, err := (sigmaevals.AnswerScorer{Mode: sigmaevals.MatchContains}).Score(context.Background(), sigmaevals.ScoreInput{
		Case: sigmaevals.Case{ID: "route", Expected: sigmaevals.Expected{
			Answers:         []string{"approved route"},
			NegativeAnswers: []string{"blocked route"},
		}},
		Output: "This mentions the blocked route, not the approved route.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if score.Passed || score.Score != 0 {
		t.Fatalf("score = %+v, want failing zero score", score)
	}
}

func TestJSONMatchScorerComparesStructuredOutput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		output string
		passed bool
	}{
		{name: "same structure different field order", output: `{"count":2,"answer":"yes"}`, passed: true},
		{name: "different value", output: `{"count":3,"answer":"yes"}`, passed: false},
		{name: "invalid json", output: `{"count":`, passed: false},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			score, err := sigmaevals.JSONMatchScorer{}.Score(context.Background(), sigmaevals.ScoreInput{
				Case:   sigmaevals.Case{ID: "json", Expected: sigmaevals.Expected{JSON: map[string]any{"answer": "yes", "count": 2}}},
				Output: tt.output,
			})
			if err != nil {
				t.Fatal(err)
			}
			if score.Passed != tt.passed {
				t.Fatalf("score = %+v, want passed=%v", score, tt.passed)
			}
		})
	}
}

func TestTokenF1ScorerUsesBestAlias(t *testing.T) {
	t.Parallel()

	score, err := sigmaevals.TokenF1Scorer{}.Score(context.Background(), sigmaevals.ScoreInput{
		Case:   sigmaevals.Case{ID: "qa", Expected: sigmaevals.Expected{Answers: []string{"quick brown fox", "fox"}}},
		Output: "The fox.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if score.Score != 1 {
		t.Fatalf("score = %.3f, want best alias F1 of 1", score.Score)
	}
}
