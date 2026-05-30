// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/fireworks"
	"github.com/wintermi/sigma/provider/opencode"
)

type liveOutput struct {
	StartedAt   time.Time            `json:"startedAt"`
	EndedAt     time.Time            `json:"endedAt"`
	Models      []string             `json:"models"`
	Judge       string               `json:"judge"`
	Needle      sigmaevals.RunResult `json:"needle"`
	ToolCalling []toolRunResult      `json:"toolCalling"`
	Errors      []string             `json:"errors,omitempty"`
}

type toolRunResult struct {
	Model            string                 `json:"model"`
	Provider         sigma.ProviderID       `json:"provider"`
	ToolCallScore    sigmaevals.Score       `json:"toolCallScore"`
	FinalAnswerScore sigmaevals.Score       `json:"finalAnswerScore"`
	JudgeScore       sigmaevals.Score       `json:"judgeScore"`
	FirstMessage     sigma.AssistantMessage `json:"firstMessage"`
	FinalMessage     sigma.AssistantMessage `json:"finalMessage"`
	FinalOutput      string                 `json:"finalOutput,omitempty"`
	Error            string                 `json:"error,omitempty"`
	DurationMS       int64                  `json:"durationMs"`
}

func main() {
	outDir := flag.String("out", "runs/live", "directory for summary.json")
	judgeModelID := flag.String("judge", "gpt-5.5", "OpenCode Zen judge model id")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()

	if err := requireEnv("FIREWORKS_API_KEY", "OPENCODE_API_KEY"); err != nil {
		fatal(err)
	}

	client, targets, judge, err := liveClient(*judgeModelID)
	if err != nil {
		fatal(err)
	}

	startedAt := time.Now().UTC()
	result := liveOutput{
		StartedAt: startedAt,
		Models:    modelLabels(targets),
		Judge:     modelLabel(judge),
	}

	needle, err := runNeedle(ctx, client, targets, judge)
	if err != nil {
		result.Errors = append(result.Errors, err.Error())
	} else {
		result.Needle = needle
	}

	toolResults := make([]toolRunResult, 0, len(targets))
	for _, target := range targets {
		toolResult := runToolCalling(ctx, client, target, judge)
		toolResults = append(toolResults, toolResult)
		if toolResult.Error != "" {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %s", toolResult.Model, toolResult.Error))
		}
	}
	result.ToolCalling = toolResults
	result.EndedAt = time.Now().UTC()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}
	path := filepath.Join(*outDir, "summary.json")
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		fatal(err)
	}

	fmt.Printf("wrote %s\n", path)
	fmt.Printf("needle: %d/%d passed, errors=%d\n", result.Needle.Summary.Passed, result.Needle.Summary.Total, result.Needle.Summary.Errors)
	passedTools := 0
	for _, tool := range result.ToolCalling {
		if tool.Error == "" && tool.ToolCallScore.Passed && tool.FinalAnswerScore.Passed && tool.JudgeScore.Passed {
			passedTools++
		}
	}
	fmt.Printf("tool calling: %d/%d passed\n", passedTools, len(result.ToolCalling))
	if len(result.Errors) > 0 {
		fmt.Printf("errors: %d\n", len(result.Errors))
		os.Exit(1)
	}
}

func liveClient(judgeModelID string) (*sigma.Client, []sigma.Model, sigma.Model, error) {
	registry := sigma.DefaultRegistry()
	if err := fireworks.Register(registry); err != nil {
		return nil, nil, sigma.Model{}, err
	}
	if err := opencode.RegisterDefault(registry); err != nil {
		return nil, nil, sigma.Model{}, err
	}
	judge := sigma.Model{
		ID:               sigma.ModelID(judgeModelID),
		Provider:         sigma.ProviderOpenCode,
		API:              sigma.APIOpenAICompletions,
		Name:             judgeModelID + " high-reasoning judge",
		ContextWindow:    400000,
		MaxOutputTokens:  4096,
		SupportedInputs:  []sigma.ContentBlockType{sigma.ContentBlockText},
		SupportsThinking: true,
		ThinkingLevels:   []sigma.ThinkingLevel{sigma.ThinkingLevelLow, sigma.ThinkingLevelMedium, sigma.ThinkingLevelHigh},
		DefaultTransport: sigma.TransportSSE,
		ProviderMetadata: map[string]any{
			sigma.MetadataAPIKeyEnvVars: []string{"OPENCODE_API_KEY"},
			"opencodeAPI":               string(sigma.APIOpenAIResponses),
		},
	}
	if err := registry.RegisterModel(judge, sigma.WithOverride()); err != nil {
		return nil, nil, sigma.Model{}, err
	}

	targets := make([]sigma.Model, 0, 3)
	for _, ref := range []sigma.ModelRef{
		{Provider: sigma.ProviderFireworks, ID: "accounts/fireworks/routers/kimi-k2p6-turbo"},
		{Provider: sigma.ProviderOpenCodeGo, ID: "kimi-k2.6"},
		{Provider: sigma.ProviderOpenCode, ID: "kimi-k2.6"},
	} {
		model, ok := registry.Model(ref.Provider, ref.ID)
		if !ok {
			return nil, nil, sigma.Model{}, fmt.Errorf("model not found: %s/%s", ref.Provider, ref.ID)
		}
		targets = append(targets, model)
	}
	return sigma.NewClient(sigma.WithRegistry(registry)), targets, judge, nil
}

func runNeedle(ctx context.Context, client *sigma.Client, targets []sigma.Model, judge sigma.Model) (sigmaevals.RunResult, error) {
	suite := sigmaevals.Suite{
		Name:         "Live Needle Retrieval",
		Version:      "v1",
		SystemPrompt: "Find the requested value in the context. Answer with only the value.",
		Cases: []sigmaevals.Case{{
			ID:    "needle-deployment-ticket",
			Input: longNeedlePrompt(),
			Expected: sigmaevals.Expected{
				Answers: []string{"TICKET-7429", "7429"},
				Rubric:  "The answer must identify the deployment ticket hidden in the context and must not use a distractor value.",
			},
			Tags: []string{"needle", "reasoning", "live"},
		}},
	}
	rubric, err := sigmaevals.PresetRubric(sigmaevals.PresetFactualCorrectness, "Score only the visible final answer. The correct value is TICKET-7429.")
	if err != nil {
		return sigmaevals.RunResult{}, err
	}
	return sigmaevals.NewRunner(client).Run(ctx, sigmaevals.RunSpec{
		Suite:  suite,
		Models: targets,
		Scorers: []sigmaevals.Scorer{
			sigmaevals.AnswerScorer{Mode: sigmaevals.MatchContains},
			sigmaevals.RubricJudgeScorer{Client: client, JudgeModel: judge, Rubric: rubric, JudgeOptions: judgeOptions()},
		},
		Options:     []sigma.Option{sigma.WithReasoningLevel(sigma.ThinkingLevelHigh), sigma.WithMaxTokens(256)},
		Concurrency: 1,
	})
}

func runToolCalling(ctx context.Context, client *sigma.Client, target sigma.Model, judge sigma.Model) toolRunResult {
	startedAt := time.Now()
	out := toolRunResult{Model: modelLabel(target), Provider: target.Provider}
	toolCase := sigmaevals.Case{
		ID:    "lookup-deployment-ticket",
		Input: "Use the lookup_fact tool to retrieve the deployment_ticket, then answer with only the ticket value.",
		Tools: []sigma.Tool{{
			Name:        "lookup_fact",
			Description: "Look up a named fact from the workspace state.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"key": map[string]any{"type": "string", "enum": []string{"deployment_ticket"}},
				},
				"required":             []string{"key"},
				"additionalProperties": false,
			},
		}},
		Expected: sigmaevals.Expected{
			Answers:   []string{"TICKET-7429", "7429"},
			ToolCalls: []sigmaevals.ExpectedToolCall{{Name: "lookup_fact"}},
			Rubric:    "The assistant must call lookup_fact before answering and the final answer must be TICKET-7429.",
		},
	}

	first, err := client.Complete(ctx, target, sigma.Request{
		SystemPrompt: "You are an agent. Use available tools when the user asks you to look up workspace state. Do not guess.",
		Messages:     []sigma.Message{sigma.UserText(toolCase.Input)},
		Tools:        toolCase.Tools,
	}, append(toolOptions(), sigma.WithReasoningLevel(sigma.ThinkingLevelHigh), sigma.WithMaxTokens(512))...)
	out.FirstMessage = first
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}

	toolScore, err := sigmaevals.ToolCallScorer{}.Score(ctx, sigmaevals.ScoreInput{Case: toolCase, Model: target, Message: first})
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	out.ToolCallScore = toolScore
	calls := sigmaevals.AssistantToolCalls(first)
	if len(calls) == 0 {
		out.Error = "model did not request a tool call"
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}

	call := calls[0]
	messages := []sigma.Message{
		sigma.UserText(toolCase.Input),
		{Role: sigma.RoleAssistant, Content: first.Content, Provider: first.Provider, API: target.API, Model: first.Model, StopReason: first.StopReason},
		{Role: sigma.RoleTool, ToolCallID: call.ID, ToolName: call.Name, Content: []sigma.ContentBlock{sigma.Text("TICKET-7429")}},
	}
	final, err := client.Complete(ctx, target, sigma.Request{
		SystemPrompt: "Use the tool result to answer with only the ticket value.",
		Messages:     messages,
	}, sigma.WithReasoningLevel(sigma.ThinkingLevelHigh), sigma.WithMaxTokens(128))
	out.FinalMessage = final
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	finalOutput, err := sigmaevals.AssistantText(final)
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	out.FinalOutput = finalOutput

	answerScore, err := sigmaevals.AnswerScorer{Mode: sigmaevals.MatchContains}.Score(ctx, sigmaevals.ScoreInput{Case: toolCase, Model: target, Output: finalOutput, Message: final})
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	out.FinalAnswerScore = answerScore
	rubric, err := sigmaevals.PresetRubric(sigmaevals.PresetToolCall, "The expected tool is lookup_fact and the expected final value is TICKET-7429.")
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	judgeScore, err := (sigmaevals.RubricJudgeScorer{Client: client, JudgeModel: judge, Rubric: rubric, JudgeOptions: judgeOptions()}).Score(ctx, sigmaevals.ScoreInput{
		Suite:   sigmaevals.Suite{Name: "Live Tool Calling", DataType: sigmaevals.EvalDataFullTrace},
		Case:    withToolTrace(toolCase, first, final),
		Model:   target,
		Output:  finalOutput,
		Message: final,
	})
	if err != nil {
		out.Error = err.Error()
		out.DurationMS = time.Since(startedAt).Milliseconds()
		return out
	}
	out.JudgeScore = judgeScore
	out.DurationMS = time.Since(startedAt).Milliseconds()
	return out
}

func withToolTrace(c sigmaevals.Case, first sigma.AssistantMessage, final sigma.AssistantMessage) sigmaevals.Case {
	c.DataType = sigmaevals.EvalDataFullTrace
	c.Trace = sigmaevals.Trace{Messages: []sigma.Message{
		{Role: sigma.RoleAssistant, Content: first.Content, Provider: first.Provider, Model: first.Model, StopReason: first.StopReason},
		{Role: sigma.RoleAssistant, Content: final.Content, Provider: final.Provider, Model: final.Model, StopReason: final.StopReason},
	}}
	return c
}

func toolOptions() []sigma.Option {
	parallel := false
	return []sigma.Option{sigma.WithOpenAIOptions(sigma.OpenAIOptions{
		ToolChoice:        map[string]any{"type": "function", "function": map[string]any{"name": "lookup_fact"}},
		ParallelToolCalls: &parallel,
	})}
}

func judgeOptions() []sigma.Option {
	return []sigma.Option{sigma.WithReasoningLevel(sigma.ThinkingLevelHigh), sigma.WithMaxTokens(1024)}
}

func longNeedlePrompt() string {
	parts := []string{
		"Context 001: The calendar owner is Mira.",
		"Context 002: The staging region is us-central1.",
		"Context 003: The archive bucket is blue-archive-17.",
		"Context 004: The incident colour is amber.",
		"Context 005: The reviewer is not assigned yet.",
		"Context 006: The deployment ticket is TICKET-7429.",
		"Context 007: The fallback region is europe-west4.",
		"Context 008: The lunch order is noodles.",
		"Context 009: The previous ticket TICKET-1001 is obsolete.",
		"Question: What is the deployment ticket?",
	}
	return strings.Join(parts, "\n")
}

func requireEnv(names ...string) error {
	missing := make([]string, 0)
	for _, name := range names {
		if os.Getenv(name) == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing required environment variables: %s", strings.Join(missing, ", "))
	}
	return nil
}

func modelLabels(models []sigma.Model) []string {
	labels := make([]string, 0, len(models))
	for _, model := range models {
		labels = append(labels, modelLabel(model))
	}
	return labels
}

func modelLabel(model sigma.Model) string {
	if model.Name != "" {
		return string(model.Provider) + "/" + model.Name
	}
	return string(model.Provider) + "/" + string(model.ID)
}

func fatal(err error) {
	if err == nil || errors.Is(err, context.Canceled) {
		return
	}
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
