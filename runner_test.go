// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestRunnerRunsCasesAcrossModel(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage("Bonjour"),
		textMessage("Cedar"),
	}}
	runner := sigmaevals.NewRunner(client)
	model := sigma.Model{ID: "model-a", Provider: "test", Name: "model-a"}

	result, err := runner.Run(context.Background(), sigmaevals.RunSpec{
		Suite: sigmaevals.Suite{
			Name:         "smoke",
			SystemPrompt: "Answer directly.",
			Cases: []sigmaevals.Case{
				{ID: "translate", Input: "Translate hello to French.", Expected: sigmaevals.Expected{Answers: []string{"Bonjour"}}},
				{ID: "codename", Input: "What is the deployment codename?", Expected: sigmaevals.Expected{Answers: []string{"Cedar", "Cedar Node"}}},
			},
		},
		Models:      []sigma.Model{model},
		Concurrency: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Total != 2 || result.Summary.Passed != 2 || result.Summary.Errors != 0 {
		t.Fatalf("summary = %+v, want two passing results", result.Summary)
	}
	if len(result.Results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(result.Results))
	}
	if result.Results[0].Request.SystemPrompt != "Answer directly." {
		t.Fatalf("system prompt = %q, want inherited suite prompt", result.Results[0].Request.SystemPrompt)
	}
	if len(client.requests) != 2 {
		t.Fatalf("recorded requests = %d, want 2", len(client.requests))
	}
}

func TestRunnerRecordsProviderErrorsWithoutDroppingOtherCases(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{textMessage("Bonjour")}}
	result, err := sigmaevals.NewRunner(client).Run(context.Background(), sigmaevals.RunSpec{
		Suite: sigmaevals.Suite{Name: "provider-error", Cases: []sigmaevals.Case{
			{ID: "a", Input: "Translate hello to French.", Expected: sigmaevals.Expected{Answers: []string{"Bonjour"}}},
			{ID: "b", Input: "Translate goodbye to French.", Expected: sigmaevals.Expected{Answers: []string{"Au revoir"}}},
		}},
		Models:      []sigma.Model{{ID: "model-a", Provider: "test", Name: "model-a"}},
		Concurrency: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Total != 2 || result.Summary.Passed != 1 || result.Summary.Errors != 1 {
		t.Fatalf("summary = %+v, want one pass and one recorded provider error", result.Summary)
	}
	if result.Results[1].Error == "" {
		t.Fatalf("second result = %+v, want provider error recorded", result.Results[1])
	}
}

type scriptedClient struct {
	mu        sync.Mutex
	responses []sigma.AssistantMessage
	requests  []sigma.Request
}

func (c *scriptedClient) Complete(_ context.Context, _ sigma.Model, request sigma.Request, _ ...sigma.Option) (sigma.AssistantMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, request)
	if len(c.responses) == 0 {
		return sigma.AssistantMessage{}, errors.New("no scripted response")
	}
	response := c.responses[0]
	c.responses = c.responses[1:]
	return response, nil
}

func textMessage(text string) sigma.AssistantMessage {
	return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text(text)}}
}
