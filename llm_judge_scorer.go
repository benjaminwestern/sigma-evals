// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"strings"

	"github.com/wintermi/sigma"
)

// LLMJudgeScorer adapts Evaluator.Judge into the Scorer interface.
type LLMJudgeScorer struct {
	Client       Completer
	JudgeModel   sigma.Model
	Mode         Mode
	Rubric       string
	JudgeOptions []sigma.Option
}

// Name implements Scorer.
func (s LLMJudgeScorer) Name() string {
	mode := normalizeMode(s.Mode)
	return "llm_judge_" + string(mode)
}

// Score implements Scorer.
func (s LLMJudgeScorer) Score(ctx context.Context, input ScoreInput) (Score, error) {
	rubric := strings.TrimSpace(s.Rubric)
	if rubric == "" {
		rubric = input.Case.Expected.Rubric
	}

	groundTruth := input.Case.Expected.Output
	if groundTruth == "" && len(input.Case.Expected.Answers) > 0 {
		groundTruth = strings.Join(input.Case.Expected.Answers, "\n")
	}

	result, err := NewEvaluator(s.Client).Judge(ctx, JudgeInput{
		Input:        input.Case.Input,
		TargetOutput: input.Output,
		GroundTruth:  groundTruth,
		Rubric:       rubric,
		JudgeModel:   s.JudgeModel,
		Mode:         s.Mode,
		JudgeOptions: s.JudgeOptions,
	})
	if err != nil {
		return Score{}, err
	}

	details := map[string]any{
		"mode":           result.Mode,
		"rawJudgeOutput": result.RawJudgeOutput,
	}
	if result.JSON != "" {
		details["json"] = result.JSON
	}
	if len(result.Logprobs) > 0 {
		details["logprobs"] = result.Logprobs
	}
	return Score{
		Name:      s.Name(),
		Score:     result.Score,
		Passed:    result.Passed,
		Rationale: result.Rationale,
		Details:   details,
	}, nil
}
