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

func TestEvaluatorUsesTargetCompleterForTargetAndJudge(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage("Bonjour"),
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
	}}
	result, err := sigmaevals.NewTargetEvaluator(completer).Evaluate(context.Background(), sigmaevals.EvaluateInput{
		Input:        "Translate hello to French.",
		GroundTruth:  "Bonjour",
		Rubric:       "Grade exactness only.",
		Target:       sigmaevals.Target{Provider: "agent-runtime", ModelID: "worker"},
		Judge:        sigmaevals.Target{Provider: "agent-runtime", ModelID: "judge"},
		TargetPrompt: "Answer the task.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Passed || result.TargetOutput != "Bonjour" {
		t.Fatalf("result = %+v, want target output judged through target completer", result)
	}
	if len(completer.requests) != 2 {
		t.Fatalf("requests = %d, want target and judge calls", len(completer.requests))
	}
	if completer.requests[0].Target.ModelID != "worker" || completer.requests[1].Target.ModelID != "judge" {
		t.Fatalf("requests = %+v, want target then judge targets", completer.requests)
	}
}

func TestEvaluatorGEvalUsesCustomPassThreshold(t *testing.T) {
	t.Parallel()

	judgeMessage := textMessage("4")
	judgeMessage.ProviderMetadata = map[string]any{
		"logprobs": []sigmaevals.TokenLogprob{{
			Token: "4",
			TopLogprobs: []sigmaevals.TokenLogprob{
				{Token: "4", Logprob: math.Log(0.75)},
				{Token: "5", Logprob: math.Log(0.25)},
			},
		}},
	}
	result, err := sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{responses: []sigma.AssistantMessage{judgeMessage}}).Judge(context.Background(), sigmaevals.JudgeInput{
		TargetOutput:  "Bonjour",
		Judge:         sigmaevals.Target{Provider: "agent-runtime", ModelID: "judge"},
		Mode:          sigmaevals.ModeGEval,
		PassThreshold: 4.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result.Score-4.25) > 0.0001 {
		t.Fatalf("score = %.4f, want 4.25", result.Score)
	}
	if result.Passed {
		t.Fatalf("passed = true, want false with threshold 4.5")
	}
	if result.PassThreshold != 4.5 {
		t.Fatalf("pass threshold = %.2f, want 4.5", result.PassThreshold)
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
