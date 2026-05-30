// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"reflect"

	"github.com/wintermi/sigma"
)

// ToolCallScorer verifies that the assistant requested the expected tools.
type ToolCallScorer struct{}

// Name implements Scorer.
func (ToolCallScorer) Name() string { return "tool_call" }

// Score implements Scorer.
func (ToolCallScorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	expected := input.Case.Expected.ToolCalls
	if len(expected) == 0 {
		return Score{}, fmt.Errorf("case %q has no expected tool calls", input.Case.ID)
	}

	actual := append(AssistantToolCalls(input.Message), TraceToolCalls(input.Case.Trace)...)
	missing := make([]string, 0)
	for _, want := range expected {
		if !hasToolCall(actual, want) {
			missing = append(missing, want.Name)
		}
	}
	if len(missing) > 0 {
		return Score{
			Name:      "tool_call",
			Score:     0,
			Passed:    false,
			Rationale: fmt.Sprintf("missing expected tool calls: %v", missing),
			Details:   map[string]any{"actual": actual},
		}, nil
	}
	return Score{
		Name:      "tool_call",
		Score:     1,
		Passed:    true,
		Rationale: "all expected tool calls were present",
		Details:   map[string]any{"actual": actual},
	}, nil
}

// AssistantToolCalls extracts tool calls from an assistant message.
func AssistantToolCalls(message sigma.AssistantMessage) []sigma.ToolCall {
	return contentToolCalls(message.Content)
}

// TraceToolCalls extracts tool calls from a stored full trace.
func TraceToolCalls(trace Trace) []sigma.ToolCall {
	calls := make([]sigma.ToolCall, 0)
	for _, message := range trace.Messages {
		calls = append(calls, contentToolCalls(message.Content)...)
	}
	for _, event := range trace.Events {
		if event.Type != "tool_call" && event.Type != "tool-call" {
			continue
		}
		call, ok := event.Payload.(sigma.ToolCall)
		if ok {
			calls = append(calls, call)
			continue
		}
		data, err := normalizeJSONValue(event.Payload)
		if err != nil {
			continue
		}
		if values, ok := data.(map[string]any); ok {
			call := sigma.ToolCall{Name: event.Name, Arguments: values["arguments"]}
			if name, ok := values["name"].(string); ok && name != "" {
				call.Name = name
			}
			if id, ok := values["id"].(string); ok {
				call.ID = id
			}
			if call.Name != "" {
				calls = append(calls, call)
			}
		}
	}
	return calls
}

func contentToolCalls(content []sigma.ContentBlock) []sigma.ToolCall {
	calls := make([]sigma.ToolCall, 0)
	for _, block := range content {
		if block.Type != sigma.ContentBlockToolCall {
			continue
		}
		calls = append(calls, sigma.ToolCall{
			ID:                block.ToolCallID,
			Name:              block.ToolName,
			Arguments:         block.ToolArguments,
			ProviderSignature: block.ProviderSignature,
			ProviderMetadata:  block.ProviderMetadata,
		})
	}
	return calls
}

func hasToolCall(actual []sigma.ToolCall, expected ExpectedToolCall) bool {
	for _, call := range actual {
		if call.Name != expected.Name {
			continue
		}
		if expected.Arguments == nil {
			return true
		}
		actualArgs, err := normalizeJSONValue(call.Arguments)
		if err != nil {
			continue
		}
		expectedArgs, err := normalizeJSONValue(expected.Arguments)
		if err != nil {
			continue
		}
		if reflect.DeepEqual(actualArgs, expectedArgs) {
			return true
		}
	}
	return false
}
