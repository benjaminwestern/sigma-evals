// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"errors"
	"math"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestParseJSONJudgeResultRejectsExtraFields(t *testing.T) {
	t.Parallel()

	_, err := sigmaevals.ParseJSONJudgeResult(`{"score":1,"rationale":"ok","passed":true,"extra":1}`)
	if !errors.Is(err, sigmaevals.ErrInvalidJudgeResult) {
		t.Fatalf("err = %v, want ErrInvalidJudgeResult", err)
	}
}

func TestEvaluatorRepairsInvalidJSONJudgeOutput(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage("not json"),
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
	}}
	model := sigma.Model{ID: "judge", Provider: "test"}

	result, err := sigmaevals.NewEvaluator(client).Judge(context.Background(), sigmaevals.JudgeInput{
		Input:        "Translate hello to French.",
		TargetOutput: "Bonjour",
		GroundTruth:  "Bonjour",
		Rubric:       "Grade exactness only.",
		JudgeModel:   model,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Passed || result.Score != 1 || result.Rationale != "exact" {
		t.Fatalf("result = %+v, want repaired passing result", result)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want repair attempt", len(client.requests))
	}
}

func TestEvaluatorReturnsInvalidJudgeResultAfterRepairFails(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage("not json"),
		textMessage(`{"score":"bad","rationale":"still invalid","passed":true}`),
	}}
	_, err := sigmaevals.NewEvaluator(client).Judge(context.Background(), sigmaevals.JudgeInput{
		Input:        "Translate hello to French.",
		TargetOutput: "Bonjour",
		GroundTruth:  "Bonjour",
		Rubric:       "Grade exactness only.",
		JudgeModel:   sigma.Model{ID: "judge", Provider: "test"},
	})
	if !errors.Is(err, sigmaevals.ErrInvalidJudgeResult) {
		t.Fatalf("err = %v, want ErrInvalidJudgeResult", err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want initial attempt plus repair", len(client.requests))
	}
}

func TestGEvalScoreForOutputUsesVisibleScoreTokenLogprobs(t *testing.T) {
	t.Parallel()

	score, ok := sigmaevals.GEvalScoreForOutput([]sigmaevals.TokenLogprob{
		{Token: "hidden", TopLogprobs: []sigmaevals.TokenLogprob{{Token: "1", Logprob: 0}}},
		{Token: "4", TopLogprobs: []sigmaevals.TokenLogprob{{Token: "4", Logprob: math.Log(0.75)}, {Token: "5", Logprob: math.Log(0.25)}}},
	}, "4")
	if !ok {
		t.Fatal("GEvalScoreForOutput() did not find score")
	}
	if math.Abs(score-4.25) > 0.0001 {
		t.Fatalf("score = %.4f, want 4.25", score)
	}
}
