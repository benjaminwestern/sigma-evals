// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestEvaluatorEvaluateBatchRunsTargetAndJudge(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage("Bonjour"),
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
	}}
	result, err := sigmaevals.NewTargetEvaluator(completer).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:         "translation-batch",
		Target:       sigmaevals.Target{Provider: "agent", ModelID: "worker"},
		Judge:        sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Rubric:       "accuracy",
		TargetPrompt: "Answer directly.",
		Cases: []sigmaevals.JudgeCase{{
			ID:          "case-1",
			Input:       "Translate hello to French.",
			GroundTruth: "Bonjour",
			Tags:        []string{"translation"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Total != 1 || result.Summary.Passed != 1 || result.Summary.Errors != 0 {
		t.Fatalf("summary = %+v, want one passing result", result.Summary)
	}
	if len(result.Results) != 1 || result.Results[0].Score != 1 || result.Results[0].TargetOutput != "Bonjour" {
		t.Fatalf("results = %+v, want judged target output", result.Results)
	}
	if len(completer.requests) != 2 {
		t.Fatalf("requests = %d, want target and judge", len(completer.requests))
	}
	judgePrompt := completer.requests[1].Request.Messages[0].Content[0].Text
	if !strings.Contains(judgePrompt, "factual accuracy") {
		t.Fatalf("judge prompt did not resolve accuracy rubric: %s", judgePrompt)
	}
}

func TestEvaluatorEvaluateBatchGEvalRecordsCaseError(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage("Bonjour"),
		textMessage("4"),
	}}
	result, err := sigmaevals.NewTargetEvaluator(completer).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:   "g-eval-batch",
		Target: sigmaevals.Target{Provider: "agent", ModelID: "worker"},
		Judge:  sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Mode:   sigmaevals.ModeGEval,
		Cases: []sigmaevals.JudgeCase{{
			ID:          "case-1",
			Input:       "Translate hello to French.",
			GroundTruth: "Bonjour",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Errors != 1 || len(result.Results) != 1 {
		t.Fatalf("result = %+v, want one recorded case error", result)
	}
	if !strings.Contains(result.Results[0].Error, sigmaevals.ErrGEvalLogprobsRequired.Error()) {
		t.Fatalf("case error = %q, want logprobs required", result.Results[0].Error)
	}
}

func TestEvaluatorEvaluateBatchGEvalUsesJudgeLogprobs(t *testing.T) {
	t.Parallel()

	judgeMessage := textMessage("4")
	judgeMessage.ProviderMetadata = map[string]any{
		"logprobs": []sigmaevals.TokenLogprob{{
			Token: "4",
			TopLogprobs: []sigmaevals.TokenLogprob{
				{Token: "4", Logprob: math.Log(0.25)},
				{Token: "5", Logprob: math.Log(0.75)},
			},
		}},
	}
	result, err := sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage("Bonjour"),
		judgeMessage,
	}}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:   "g-eval-batch",
		Target: sigmaevals.Target{Provider: "agent", ModelID: "worker"},
		Judge:  sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Mode:   sigmaevals.ModeGEval,
		Cases: []sigmaevals.JudgeCase{{
			ID:          "case-1",
			Input:       "Translate hello to French.",
			GroundTruth: "Bonjour",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Passed != 1 || len(result.Results) != 1 {
		t.Fatalf("result = %+v, want one passing result", result)
	}
	if math.Abs(result.Results[0].Score-4.75) > 0.0001 {
		t.Fatalf("score = %.4f, want 4.75", result.Results[0].Score)
	}
}

func TestEvaluatorEvaluateBatchJudgesSavedOutputWithoutTarget(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
	}}
	result, err := sigmaevals.NewTargetEvaluator(completer).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:  "saved-output-batch",
		Judge: sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Cases: []sigmaevals.JudgeCase{{
			ID:           "case-1",
			Input:        "Translate hello to French.",
			TargetOutput: "Bonjour",
			GroundTruth:  "Bonjour",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Passed != 1 || len(completer.requests) != 1 {
		t.Fatalf("result = %+v requests=%d, want one judge-only call", result, len(completer.requests))
	}
	if completer.requests[0].Metadata["role"] != "judge" {
		t.Fatalf("request metadata = %+v, want judge-only request", completer.requests[0].Metadata)
	}
}

func TestEvaluatorEvaluateBatchRepeatsSortsAndSummarizesMixedSavedOutputs(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{
		textMessage(`{"score":0.25,"rationale":"wrong","passed":false}`),
		textMessage(`{"score":0.75,"rationale":"better","passed":true}`),
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
		textMessage(`{"score":1,"rationale":"exact","passed":true}`),
	}}
	result, err := sigmaevals.NewTargetEvaluator(completer).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:    "saved-output-batch",
		Judge:   sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Repeats: 2,
		Cases: []sigmaevals.JudgeCase{
			{ID: "case-b", TargetOutput: "wrong", GroundTruth: "right", Tags: []string{"regression"}},
			{ID: "case-a", TargetOutput: "right", GroundTruth: "right", Tags: []string{"regression"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 4 || result.Results[0].CaseID != "case-a" || result.Results[0].Repeat != 1 || result.Results[1].Repeat != 2 || result.Results[2].CaseID != "case-b" {
		t.Fatalf("results = %+v, want deterministic case-id/repeat ordering", result.Results)
	}
	if result.Summary.Total != 4 || result.Summary.Passed != 3 || result.Summary.Failed != 1 || result.Summary.ScoreCount != 4 {
		t.Fatalf("summary = %+v, want three pass and one fail", result.Summary)
	}
	if math.Abs(result.Summary.MeanScore-0.75) > 0.0001 {
		t.Fatalf("mean score = %.4f, want 0.75", result.Summary.MeanScore)
	}
	if result.ByTag["regression"].Total != 4 || result.ByModel["saved-output"].Total != 4 {
		t.Fatalf("byTag = %+v byModel = %+v, want grouped summaries", result.ByTag, result.ByModel)
	}
}

func TestEvaluatorEvaluateBatchValidatesSpec(t *testing.T) {
	t.Parallel()

	_, err := sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{})
	if err == nil {
		t.Fatal("EvaluateBatch succeeded with empty spec, want error")
	}

	_, err = sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:  "missing-target",
		Judge: sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Cases: []sigmaevals.JudgeCase{{ID: "case-1"}},
	})
	if err == nil || !strings.Contains(err.Error(), "target is required") {
		t.Fatalf("err = %v, want target required", err)
	}

	_, err = sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:   "duplicate-cases",
		Target: sigmaevals.Target{Provider: "agent", ModelID: "worker"},
		Judge:  sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Cases:  []sigmaevals.JudgeCase{{ID: "case-1"}, {ID: "case-1"}},
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("err = %v, want duplicate case id", err)
	}
}

func TestEvaluatorEvaluateBatchRecordsTargetError(t *testing.T) {
	t.Parallel()

	runtimeErr := errors.New("runtime unavailable")
	result, err := sigmaevals.NewTargetEvaluator(&recordingTargetCompleter{err: runtimeErr}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:   "error-batch",
		Target: sigmaevals.Target{Provider: "agent", ModelID: "worker"},
		Judge:  sigmaevals.Target{Provider: "agent", ModelID: "judge"},
		Cases:  []sigmaevals.JudgeCase{{ID: "case-1", Input: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Errors != 1 || !strings.Contains(result.Results[0].Error, runtimeErr.Error()) {
		t.Fatalf("result = %+v, want recorded target error", result)
	}
}
