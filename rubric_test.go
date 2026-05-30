// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"math"
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestRubricScoreRawNormalizesMixedRatingDomains(t *testing.T) {
	t.Parallel()

	rubric := sigmaevals.Rubric{Dimensions: []sigmaevals.RubricDimension{
		{Name: "Accuracy", Type: sigmaevals.RatingFiveStar, Weight: 3},
		{Name: "Safety", Type: sigmaevals.RatingPassFail, Weight: 1},
		{Name: "No Critical Failure", Type: sigmaevals.RatingPassFailCritical, Weight: 2},
	}}
	scores, err := rubric.ScoreRaw(map[string]any{"accuracy": 3, "safety": "fail", "no_critical_failure": "critical"})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(scores.Aggregate-0.25) > 0.0001 {
		t.Fatalf("Aggregate = %.3f, want weighted normalized aggregate 0.25", scores.Aggregate)
	}
	if scores.Scores["no_critical_failure"] != -1 || scores.Normalized["accuracy"] != 0.5 {
		t.Fatalf("scores = %+v, want raw critical=-1 and normalized accuracy=0.5", scores)
	}
}

func TestRubricScoreRawRejectsMissingAndInvalidScores(t *testing.T) {
	t.Parallel()

	rubric := sigmaevals.Rubric{Dimensions: []sigmaevals.RubricDimension{
		{Name: "Accuracy", Type: sigmaevals.RatingFiveStar},
		{Name: "Safety", Type: sigmaevals.RatingPassFail},
	}}
	tests := []struct {
		name string
		raw  map[string]any
	}{
		{name: "missing dimension", raw: map[string]any{"accuracy": 4}},
		{name: "five star out of range", raw: map[string]any{"accuracy": 6, "safety": "pass"}},
		{name: "unknown categorical", raw: map[string]any{"accuracy": 4, "safety": "maybe"}},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if _, err := rubric.ScoreRaw(tt.raw); err == nil {
				t.Fatalf("ScoreRaw(%v) succeeded, want error", tt.raw)
			}
		})
	}
}

func TestRubricJudgeScorerUsesCaseSpecificRubric(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage(`{"accuracy":5,"safety":"pass"}`),
	}}
	scorer := sigmaevals.RubricJudgeScorer{
		Client:     client,
		JudgeModel: sigma.Model{ID: "judge", Provider: "test"},
		Rubric: sigmaevals.Rubric{
			Name:      "quality",
			Threshold: 0.75,
			Dimensions: []sigmaevals.RubricDimension{
				{Name: "Accuracy", Type: sigmaevals.RatingFiveStar},
				{Name: "Safety", Type: sigmaevals.RatingPassFail},
			},
		},
	}
	score, err := scorer.Score(context.Background(), sigmaevals.ScoreInput{
		Case:   sigmaevals.Case{ID: "case", Input: "input", Expected: sigmaevals.Expected{Rubric: "Prefer exact answers."}},
		Output: "answer",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !score.Passed || math.Abs(score.Score-1) > 0.0001 {
		t.Fatalf("score = %+v, want passing aggregate of 1", score)
	}
	if len(client.requests) != 1 || !strings.Contains(client.requests[0].Messages[0].Content[0].Text, "Prefer exact answers.") {
		t.Fatalf("judge prompt did not include case-specific rubric: %+v", client.requests)
	}
}
