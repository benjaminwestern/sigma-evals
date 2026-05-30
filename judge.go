// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/wintermi/sigma"
)

const (
	// ModeEvaluate asks the judge for a strict JSON score object.
	ModeEvaluate Mode = "evaluate"
	// ModeGEval asks the judge for a single 1-5 score and derives the score from logprobs.
	ModeGEval Mode = "g_eval"
)

const (
	defaultRepairTurns = 1
	defaultTopLogprobs = 5
)

var (
	// ErrInvalidJudgeResult indicates the judge returned an unusable evaluation result.
	ErrInvalidJudgeResult = errors.New("invalid judge result")
	// ErrGEvalLogprobsRequired indicates G-Eval could not find score-token logprobs.
	ErrGEvalLogprobsRequired = errors.New("g-eval requires provider logprobs for score tokens 1-5")
)

// Mode identifies an LLM judge strategy.
type Mode string

// Evaluator runs target and judge model calls through Sigma.
type Evaluator struct {
	Client Completer
}

// NewEvaluator constructs an Evaluator. A nil client uses sigma.NewClient at call time.
func NewEvaluator(client Completer) *Evaluator {
	return &Evaluator{Client: client}
}

// EvaluateInput configures target generation plus judge evaluation.
type EvaluateInput struct {
	Input         string         `json:"input,omitempty"`
	GroundTruth   string         `json:"groundTruth,omitempty"`
	Rubric        string         `json:"rubric,omitempty"`
	TargetPrompt  string         `json:"targetPrompt,omitempty"`
	TargetModel   sigma.Model    `json:"targetModel"`
	JudgeModel    sigma.Model    `json:"judgeModel"`
	Mode          Mode           `json:"mode,omitempty"`
	TargetOptions []sigma.Option `json:"-"`
	JudgeOptions  []sigma.Option `json:"-"`
}

// JudgeInput configures evaluation for an already-generated target output.
type JudgeInput struct {
	Input        string         `json:"input,omitempty"`
	TargetOutput string         `json:"targetOutput"`
	GroundTruth  string         `json:"groundTruth,omitempty"`
	Rubric       string         `json:"rubric,omitempty"`
	JudgeModel   sigma.Model    `json:"judgeModel"`
	Mode         Mode           `json:"mode,omitempty"`
	JudgeOptions []sigma.Option `json:"-"`
}

// JudgeResult is the normalized score returned by an LLM judge.
type JudgeResult struct {
	Mode           Mode                   `json:"mode"`
	Input          string                 `json:"input,omitempty"`
	TargetOutput   string                 `json:"targetOutput,omitempty"`
	GroundTruth    string                 `json:"groundTruth,omitempty"`
	Score          float64                `json:"score"`
	Rationale      string                 `json:"rationale,omitempty"`
	Passed         bool                   `json:"passed"`
	JSON           string                 `json:"json,omitempty"`
	RawJudgeOutput string                 `json:"rawJudgeOutput,omitempty"`
	Logprobs       []TokenLogprob         `json:"logprobs,omitempty"`
	TargetMessage  sigma.AssistantMessage `json:"targetMessage,omitempty"`
	JudgeMessage   sigma.AssistantMessage `json:"judgeMessage,omitempty"`
}

// JSONJudgeResult is the strict JSON score shape returned by ModeEvaluate.
type JSONJudgeResult struct {
	Score     float64 `json:"score"`
	Rationale string  `json:"rationale"`
	Passed    bool    `json:"passed"`
	JSON      string  `json:"-"`
}

// Evaluate generates target output, then evaluates it with the judge model.
func (e *Evaluator) Evaluate(ctx context.Context, input EvaluateInput) (JudgeResult, error) {
	mode := normalizeMode(input.Mode)
	result := JudgeResult{Mode: mode, Input: input.Input, GroundTruth: input.GroundTruth}

	final, err := e.clientOrDefault().Complete(ctx, input.TargetModel, sigma.Request{
		Messages: []sigma.Message{sigma.UserText(formatTargetPrompt(input.TargetPrompt, input.Input))},
	}, input.TargetOptions...)
	result.TargetMessage = final
	if err != nil {
		return result, err
	}

	targetOutput, err := AssistantText(final)
	if err != nil {
		return result, err
	}
	result.TargetOutput = targetOutput

	judgeResult, err := e.Judge(ctx, JudgeInput{
		Input:        input.Input,
		TargetOutput: targetOutput,
		GroundTruth:  input.GroundTruth,
		Rubric:       input.Rubric,
		JudgeModel:   input.JudgeModel,
		Mode:         mode,
		JudgeOptions: input.JudgeOptions,
	})
	judgeResult.TargetMessage = final
	return judgeResult, err
}

// Judge evaluates an existing target output with the configured judge model.
func (e *Evaluator) Judge(ctx context.Context, input JudgeInput) (JudgeResult, error) {
	mode := normalizeMode(input.Mode)
	result := JudgeResult{Mode: mode, Input: input.Input, TargetOutput: input.TargetOutput, GroundTruth: input.GroundTruth}

	switch mode {
	case ModeEvaluate:
		return e.judgeJSON(ctx, input, result)
	case ModeGEval:
		return e.judgeGEval(ctx, input, result)
	default:
		return result, fmt.Errorf("unsupported evaluation mode %q", mode)
	}
}

// ParseJSONJudgeResult validates the strict JSON result shape used by ModeEvaluate.
func ParseJSONJudgeResult(text string) (JSONJudgeResult, error) {
	decoder := json.NewDecoder(strings.NewReader(strings.TrimSpace(text)))
	decoder.UseNumber()

	var values map[string]any
	if err := decoder.Decode(&values); err != nil {
		return JSONJudgeResult{}, fmt.Errorf("%w: decode json: %v", ErrInvalidJudgeResult, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return JSONJudgeResult{}, fmt.Errorf("%w: trailing json value", ErrInvalidJudgeResult)
	}
	if err := validateJSONJudgeResultKeys(values); err != nil {
		return JSONJudgeResult{}, err
	}

	score, ok := numberValue(values["score"])
	if !ok {
		return JSONJudgeResult{}, fmt.Errorf("%w: score must be a number", ErrInvalidJudgeResult)
	}
	rationale, ok := values["rationale"].(string)
	if !ok || strings.TrimSpace(rationale) == "" {
		return JSONJudgeResult{}, fmt.Errorf("%w: rationale must be a non-empty string", ErrInvalidJudgeResult)
	}
	passed, ok := values["passed"].(bool)
	if !ok {
		return JSONJudgeResult{}, fmt.Errorf("%w: passed must be a boolean", ErrInvalidJudgeResult)
	}
	return newJSONJudgeResult(score, rationale, passed)
}

func (e *Evaluator) judgeJSON(ctx context.Context, input JudgeInput, result JudgeResult) (JudgeResult, error) {
	prompt := formatJudgePrompt(input, ModeEvaluate)
	options := appendOptions(input.JudgeOptions, withStructuredOutput(input.JudgeModel, scoreResponseFormat()))

	var lastErr error
	for attempt := 0; attempt <= defaultRepairTurns; attempt++ {
		if attempt > 0 {
			prompt = repairPrompt(formatJudgePrompt(input, ModeEvaluate), result.RawJudgeOutput, lastErr)
		}
		final, err := e.clientOrDefault().Complete(ctx, input.JudgeModel, sigma.Request{
			Messages: []sigma.Message{sigma.UserText(prompt)},
		}, options...)
		result.JudgeMessage = final
		if err != nil {
			return result, err
		}

		rawOutput, err := AssistantText(final)
		result.RawJudgeOutput = rawOutput
		if err != nil {
			lastErr = err
			continue
		}
		parsed, err := ParseJSONJudgeResult(rawOutput)
		if err != nil {
			lastErr = err
			continue
		}
		applyJSONJudgeResult(&result, parsed)
		return result, nil
	}
	return result, fmt.Errorf("parse evaluation result: %w", lastErr)
}

func (e *Evaluator) judgeGEval(ctx context.Context, input JudgeInput, result JudgeResult) (JudgeResult, error) {
	final, err := e.clientOrDefault().Complete(ctx, input.JudgeModel, sigma.Request{
		Messages: []sigma.Message{sigma.UserText(formatJudgePrompt(input, ModeGEval))},
	}, appendOptions(
		input.JudgeOptions,
		sigma.WithReasoningLevel(sigma.ThinkingLevelOff),
		withOpenAILogprobs(defaultTopLogprobs),
	)...)
	result.JudgeMessage = final
	if err != nil {
		return result, err
	}
	rawOutput, err := AssistantText(final)
	if err != nil {
		return result, ErrGEvalLogprobsRequired
	}
	result.RawJudgeOutput = rawOutput

	logprobs, ok := TokenLogprobsFromMetadata(final.ProviderMetadata)
	if !ok {
		return result, ErrGEvalLogprobsRequired
	}
	score, ok := GEvalScoreForOutput(logprobs, rawOutput)
	if !ok {
		return result, ErrGEvalLogprobsRequired
	}
	parsed, err := newJSONJudgeResult(score, "G-Eval logarithmic probability evaluation", score >= 3)
	if err != nil {
		return result, err
	}
	result.Logprobs = logprobs
	applyJSONJudgeResult(&result, parsed)
	return result, nil
}

func (e *Evaluator) clientOrDefault() Completer {
	if e == nil || e.Client == nil {
		return sigma.NewClient()
	}
	return e.Client
}

func normalizeMode(mode Mode) Mode {
	if mode == "" {
		return ModeEvaluate
	}
	return mode
}

func formatTargetPrompt(prompt string, input string) string {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return input
	}
	if strings.TrimSpace(input) == "" {
		return prompt
	}
	return prompt + "\n\nInput:\n" + input
}

func formatJudgePrompt(input JudgeInput, mode Mode) string {
	var b strings.Builder
	if strings.TrimSpace(input.Rubric) != "" {
		b.WriteString(input.Rubric)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(input.Input) != "" {
		b.WriteString("Input Context:\n")
		b.WriteString(input.Input)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(input.GroundTruth) != "" {
		b.WriteString("Expected Ground Truth:\n")
		b.WriteString(input.GroundTruth)
		b.WriteString("\n\n")
	}
	b.WriteString("Target Output:\n")
	b.WriteString(input.TargetOutput)
	b.WriteString("\n\n")
	switch mode {
	case ModeGEval:
		b.WriteString("Output ONLY a single integer score between 1 and 5. Do not output anything else. No explanation, no JSON, just the number.")
	default:
		b.WriteString("Return exactly one JSON object matching this shape and no surrounding prose: ")
		b.WriteString(`{"score":0.0,"rationale":"...","passed":true}`)
	}
	return b.String()
}

func repairPrompt(originalPrompt string, rawOutput string, err error) string {
	var b strings.Builder
	b.WriteString(originalPrompt)
	b.WriteString("\n\nPrevious response was invalid")
	if err != nil {
		b.WriteString(": ")
		b.WriteString(err.Error())
	}
	if strings.TrimSpace(rawOutput) != "" {
		b.WriteString("\n\nPrevious response:\n")
		b.WriteString(rawOutput)
	}
	b.WriteString("\n\nReply again with exactly one valid JSON object matching the requested shape and no extra prose.")
	return b.String()
}

func appendOptions(base []sigma.Option, extra ...sigma.Option) []sigma.Option {
	out := make([]sigma.Option, 0, len(base)+len(extra))
	out = append(out, base...)
	out = append(out, extra...)
	return out
}

func scoreResponseFormat() map[string]any {
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "evaluation_result",
			"strict": true,
			"schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"score":     map[string]any{"type": "number"},
					"rationale": map[string]any{"type": "string"},
					"passed":    map[string]any{"type": "boolean"},
				},
				"required":             []string{"score", "rationale", "passed"},
				"additionalProperties": false,
			},
		},
	}
}

func validateJSONJudgeResultKeys(values map[string]any) error {
	required := map[string]struct{}{
		"score":     {},
		"rationale": {},
		"passed":    {},
	}
	if len(values) != len(required) {
		return fmt.Errorf("%w: expected score, rationale, and passed only", ErrInvalidJudgeResult)
	}
	for key := range values {
		if _, ok := required[key]; !ok {
			return fmt.Errorf("%w: unexpected field %q", ErrInvalidJudgeResult, key)
		}
	}
	return nil
}

func numberValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case json.Number:
		value, err := typed.Float64()
		return value, err == nil
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	default:
		return 0, false
	}
}

func newJSONJudgeResult(score float64, rationale string, passed bool) (JSONJudgeResult, error) {
	encoded, err := json.Marshal(map[string]any{
		"score":     score,
		"rationale": rationale,
		"passed":    passed,
	})
	if err != nil {
		return JSONJudgeResult{}, err
	}
	return JSONJudgeResult{Score: score, Rationale: rationale, Passed: passed, JSON: string(encoded)}, nil
}

func applyJSONJudgeResult(result *JudgeResult, parsed JSONJudgeResult) {
	result.Score = parsed.Score
	result.Rationale = parsed.Rationale
	result.Passed = parsed.Passed
	result.JSON = parsed.JSON
}
