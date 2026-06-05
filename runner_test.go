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
	"time"

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

func TestRunnerClassifiesProviderErrors(t *testing.T) {
	t.Parallel()

	providerErr := sigma.NewProviderError(
		sigma.ProviderOpenAI,
		sigma.APIOpenAIResponses,
		"gpt-5.3",
		429,
		"req_123",
		2*time.Second,
		[]byte(`{"error":{"code":"rate_limit_error","message":"slow down"}}`),
		nil,
	)
	client := &scriptedClient{errors: []error{providerErr}}
	result, err := sigmaevals.NewRunner(client).Run(context.Background(), sigmaevals.RunSpec{
		Suite:  sigmaevals.Suite{Name: "classified-error", Cases: []sigmaevals.Case{{ID: "a", Input: "hello"}}},
		Models: []sigma.Model{{ID: "gpt-5.3", Provider: sigma.ProviderOpenAI}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(result.Results))
	}
	details := result.Results[0].ErrorDetails
	if details == nil {
		t.Fatalf("error details missing from result: %+v", result.Results[0])
	}
	if details.Class != sigma.ErrorClassRateLimited || !details.Retryable || details.RetryAfterMS != 2000 || details.StatusCode != 429 {
		t.Fatalf("error details = %+v, want rate-limited retryable details", details)
	}
}

type scriptedClient struct {
	mu        sync.Mutex
	responses []sigma.AssistantMessage
	errors    []error
	requests  []sigma.Request
}

func (c *scriptedClient) Complete(_ context.Context, _ sigma.Model, request sigma.Request, _ ...sigma.Option) (sigma.AssistantMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests = append(c.requests, request)
	if len(c.errors) > 0 {
		err := c.errors[0]
		c.errors = c.errors[1:]
		return sigma.AssistantMessage{}, err
	}
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
