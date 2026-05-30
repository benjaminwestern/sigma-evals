// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/wintermi/sigma"
)

// PairwiseWinner is the result of a pairwise judge comparison.
type PairwiseWinner string

const (
	// PairwiseWinnerA means answer A won.
	PairwiseWinnerA PairwiseWinner = "A"
	// PairwiseWinnerB means answer B won.
	PairwiseWinnerB PairwiseWinner = "B"
	// PairwiseWinnerTie means the judge considered answers equivalent.
	PairwiseWinnerTie PairwiseWinner = "tie"
	// PairwiseWinnerInconsistent means swapped-order judgments disagreed.
	PairwiseWinnerInconsistent PairwiseWinner = "inconsistent"
)

// PairwiseJudgeInput configures a pairwise LLM judge. The judge is called twice:
// A/B and B/A, then the swapped result is mapped back to original labels.
type PairwiseJudgeInput struct {
	Input        string         `json:"input,omitempty"`
	AnswerA      string         `json:"answerA"`
	AnswerB      string         `json:"answerB"`
	Reference    string         `json:"reference,omitempty"`
	Rubric       string         `json:"rubric,omitempty"`
	Judge        Target         `json:"judge,omitempty"`
	JudgeModel   sigma.Model    `json:"judgeModel"`
	JudgeOptions []sigma.Option `json:"-"`
}

// PairwiseJudgeResult records both judge orders and the resolved winner.
type PairwiseJudgeResult struct {
	Winner       PairwiseWinner       `json:"winner"`
	FirstOrder   PairwiseSingleResult `json:"firstOrder"`
	SwappedOrder PairwiseSingleResult `json:"swappedOrder"`
	Consistent   bool                 `json:"consistent"`
}

// PairwiseSingleResult records one pairwise judge call.
type PairwiseSingleResult struct {
	Winner         PairwiseWinner         `json:"winner"`
	Rationale      string                 `json:"rationale,omitempty"`
	RawJudgeOutput string                 `json:"rawJudgeOutput,omitempty"`
	JudgeMessage   sigma.AssistantMessage `json:"judgeMessage,omitempty"`
}

// PairwiseJudge compares two answers with a swapped-order bias check.
func (e *Evaluator) PairwiseJudge(ctx context.Context, input PairwiseJudgeInput) (PairwiseJudgeResult, error) {
	first, err := e.pairwiseJudgeOnce(ctx, input, input.AnswerA, input.AnswerB)
	if err != nil {
		return PairwiseJudgeResult{}, err
	}
	swapped, err := e.pairwiseJudgeOnce(ctx, input, input.AnswerB, input.AnswerA)
	if err != nil {
		return PairwiseJudgeResult{}, err
	}
	mappedSwapped := mapSwappedWinner(swapped.Winner)
	winner := first.Winner
	consistent := first.Winner == mappedSwapped
	if !consistent {
		winner = PairwiseWinnerInconsistent
	}
	return PairwiseJudgeResult{Winner: winner, FirstOrder: first, SwappedOrder: swapped, Consistent: consistent}, nil
}

func (e *Evaluator) pairwiseJudgeOnce(ctx context.Context, input PairwiseJudgeInput, answerA string, answerB string) (PairwiseSingleResult, error) {
	judgeTarget := targetWithModelFallback(input.Judge, input.JudgeModel)
	judgeResult, err := e.completeTarget(ctx, judgeTarget, sigma.Request{
		Messages: []sigma.Message{sigma.UserText(formatPairwisePrompt(input, answerA, answerB))},
	}, appendOptions(input.JudgeOptions, withStructuredOutput(pairwiseResponseFormat())), map[string]any{"role": "judge", "mode": "pairwise"})
	result := PairwiseSingleResult{JudgeMessage: judgeResult.Message}
	if err != nil {
		return result, err
	}
	rawOutput, err := targetResultText(judgeResult)
	result.RawJudgeOutput = rawOutput
	if err != nil {
		return result, err
	}
	parsed, err := ParsePairwiseJudgeOutput(rawOutput)
	if err != nil {
		return result, err
	}
	result.Winner = parsed.Winner
	result.Rationale = parsed.Rationale
	return result, nil
}

// ParsePairwiseJudgeOutput validates pairwise judge JSON.
func ParsePairwiseJudgeOutput(text string) (PairwiseSingleResult, error) {
	var parsed struct {
		Winner    string `json:"winner"`
		Rationale string `json:"rationale"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &parsed); err != nil {
		return PairwiseSingleResult{}, err
	}
	winner := PairwiseWinner(strings.ToUpper(strings.TrimSpace(parsed.Winner)))
	if strings.EqualFold(parsed.Winner, string(PairwiseWinnerTie)) {
		winner = PairwiseWinnerTie
	}
	switch winner {
	case PairwiseWinnerA, PairwiseWinnerB, PairwiseWinnerTie:
		return PairwiseSingleResult{Winner: winner, Rationale: parsed.Rationale, RawJudgeOutput: text}, nil
	default:
		return PairwiseSingleResult{}, fmt.Errorf("invalid pairwise winner %q", parsed.Winner)
	}
}

func formatPairwisePrompt(input PairwiseJudgeInput, answerA string, answerB string) string {
	var b strings.Builder
	if strings.TrimSpace(input.Rubric) != "" {
		b.WriteString(input.Rubric)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(input.Input) != "" {
		b.WriteString("Input:\n")
		b.WriteString(input.Input)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(input.Reference) != "" {
		b.WriteString("Reference answer:\n")
		b.WriteString(input.Reference)
		b.WriteString("\n\n")
	}
	b.WriteString("Answer A:\n")
	b.WriteString(answerA)
	b.WriteString("\n\nAnswer B:\n")
	b.WriteString(answerB)
	b.WriteString("\n\nReturn exactly one JSON object: {\"winner\":\"A\",\"rationale\":\"...\"}. winner must be A, B, or tie.")
	return b.String()
}

func pairwiseResponseFormat() map[string]any {
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "pairwise_judgment",
			"strict": true,
			"schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"winner":    map[string]any{"type": "string", "enum": []string{"A", "B", "tie"}},
					"rationale": map[string]any{"type": "string"},
				},
				"required":             []string{"winner", "rationale"},
				"additionalProperties": false,
			},
		},
	}
}

func mapSwappedWinner(winner PairwiseWinner) PairwiseWinner {
	switch winner {
	case PairwiseWinnerA:
		return PairwiseWinnerB
	case PairwiseWinnerB:
		return PairwiseWinnerA
	default:
		return winner
	}
}
