// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"errors"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestParseTargetAcceptsSDKTargetForms(t *testing.T) {
	t.Parallel()

	tests := []string{
		"runtime=family/model",
		"runtime/family/model",
		"runtime:family/model",
	}
	for _, raw := range tests {
		raw := raw
		t.Run(raw, func(t *testing.T) {
			t.Parallel()
			target, err := sigmaevals.ParseTarget(raw)
			if err != nil {
				t.Fatal(err)
			}
			if string(target.Provider) != "runtime" || string(target.ModelID) != "family/model" {
				t.Fatalf("target = %+v, want generic provider and nested model id", target)
			}
		})
	}

	if _, err := sigmaevals.ParseTarget("model-without-provider"); err == nil {
		t.Fatal("ParseTarget accepted ambiguous target without provider")
	}
}

func TestTargetRunnerUsesTargetCompleterContract(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{textMessage("Cedar")}}
	result, err := sigmaevals.NewTargetRunner(completer).Run(context.Background(), sigmaevals.TargetRunSpec{
		Suite: sigmaevals.Suite{Name: "sdk", Cases: []sigmaevals.Case{{
			ID:       "codename",
			Input:    "What is the codename?",
			Expected: sigmaevals.Expected{Answers: []string{"Cedar"}},
		}}},
		Targets: []sigmaevals.Target{{Provider: "runtime", ModelID: "model", Label: "runtime/model"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Passed != 1 || result.Summary.Errors != 0 {
		t.Fatalf("summary = %+v, want one passing target result", result.Summary)
	}
	if len(completer.requests) != 1 || completer.requests[0].Target.Label != "runtime/model" {
		t.Fatalf("requests = %+v, want target request recorded", completer.requests)
	}
}

func TestTargetRunnerRecordsCompleterErrors(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{err: errors.New("runtime unavailable")}
	result, err := sigmaevals.NewTargetRunner(completer).Run(context.Background(), sigmaevals.TargetRunSpec{
		Suite: sigmaevals.Suite{Name: "sdk", Cases: []sigmaevals.Case{{
			ID:       "codename",
			Input:    "What is the codename?",
			Expected: sigmaevals.Expected{Answers: []string{"Cedar"}},
		}}},
		Targets: []sigmaevals.Target{{Provider: "runtime", ModelID: "model"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Errors != 1 || result.Results[0].Error == "" {
		t.Fatalf("result = %+v, want recorded completer error", result)
	}
}

func TestRunFanoutRepeatsTargets(t *testing.T) {
	t.Parallel()

	completer := &recordingTargetCompleter{responses: []sigma.AssistantMessage{textMessage("one"), textMessage("two")}}
	result, err := sigmaevals.RunFanout(context.Background(), completer, sigmaevals.FanoutSpec{
		Request: sigma.Request{Messages: []sigma.Message{sigma.UserText("hello")}},
		Targets: []sigmaevals.Target{{Provider: "runtime", ModelID: "model"}},
		Repeats: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Total != 2 || result.Summary.Succeeded != 2 || len(result.Results) != 2 {
		t.Fatalf("result = %+v, want two successful fanout attempts", result)
	}
	if result.Results[0].Repeat != 1 || result.Results[1].Repeat != 2 {
		t.Fatalf("repeats = %d,%d, want sorted repeat numbers 1,2", result.Results[0].Repeat, result.Results[1].Repeat)
	}
	if len(completer.requests) != 2 || completer.requests[0].Request.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("requests = %+v, want original request propagated", completer.requests)
	}
}

type recordingTargetCompleter struct {
	responses []sigma.AssistantMessage
	err       error
	requests  []sigmaevals.TargetRequest
}

func (c *recordingTargetCompleter) CompleteTarget(_ context.Context, request sigmaevals.TargetRequest) (sigmaevals.TargetResult, error) {
	c.requests = append(c.requests, request)
	result := sigmaevals.TargetResult{Target: request.Target, Request: request.Request, Repeat: request.Repeat}
	if c.err != nil {
		result.Error = c.err.Error()
		return result, c.err
	}
	if len(c.responses) == 0 {
		result.Error = "no response"
		return result, nil
	}
	message := c.responses[0]
	c.responses = c.responses[1:]
	result.Message = message
	text, err := sigmaevals.AssistantText(message)
	if err == nil {
		result.Output = text
	}
	return result, nil
}
