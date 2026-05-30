// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/wintermi/sigma"
)

// ExistingOutput is a previously generated target output to score without
// calling the target model again.
type ExistingOutput struct {
	CaseID  string                 `json:"caseId"`
	Model   sigma.Model            `json:"model"`
	Repeat  int                    `json:"repeat,omitempty"`
	Output  string                 `json:"output,omitempty"`
	Message sigma.AssistantMessage `json:"message,omitempty"`
}

// ScoreExistingSpec configures scoring for already-generated outputs.
type ScoreExistingSpec struct {
	Suite   Suite            `json:"suite"`
	Outputs []ExistingOutput `json:"outputs"`
	Scorers []Scorer         `json:"-"`
}

// ScoreExisting scores existing outputs with deterministic scorers or judges.
func ScoreExisting(ctx context.Context, spec ScoreExistingSpec) (RunResult, error) {
	if len(spec.Suite.Cases) == 0 {
		return RunResult{}, fmt.Errorf("suite must contain at least one case")
	}
	if len(spec.Outputs) == 0 {
		return RunResult{}, fmt.Errorf("at least one existing output is required")
	}
	cases := make(map[string]Case, len(spec.Suite.Cases))
	for _, c := range spec.Suite.Cases {
		cases[c.ID] = c
	}
	scorers := scorersOrDefault(spec.Scorers)
	startedAt := time.Now().UTC()
	results := make([]CaseResult, 0, len(spec.Outputs))
	for _, output := range spec.Outputs {
		c, ok := cases[output.CaseID]
		if !ok {
			return RunResult{}, fmt.Errorf("unknown case id %q", output.CaseID)
		}
		started := time.Now()
		repeat := output.Repeat
		if repeat <= 0 {
			repeat = 1
		}
		result := CaseResult{
			CaseID:   c.ID,
			CaseName: c.Name,
			Tags:     append([]string(nil), c.Tags...),
			Model:    modelLabel(output.Model),
			Provider: output.Model.Provider,
			Repeat:   repeat,
			Output:   output.Output,
			Message:  output.Message,
		}
		if result.Output == "" && len(output.Message.Content) > 0 {
			text, err := AssistantText(output.Message)
			if err == nil {
				result.Output = text
			} else if len(c.Expected.ToolCalls) == 0 {
				result.Error = err.Error()
			}
		}
		if result.Error == "" {
			for _, scorer := range scorers {
				score, err := scorer.Score(ctx, ScoreInput{
					Suite:   spec.Suite,
					Case:    c,
					Model:   output.Model,
					Repeat:  repeat,
					Output:  result.Output,
					Message: output.Message,
				})
				if err != nil {
					result.Scores = append(result.Scores, Score{Name: scorer.Name(), Passed: false, Rationale: err.Error()})
					continue
				}
				if score.Name == "" {
					score.Name = scorer.Name()
				}
				result.Scores = append(result.Scores, score)
			}
		}
		result.DurationMS = time.Since(started).Milliseconds()
		results = append(results, result)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].CaseID != results[j].CaseID {
			return results[i].CaseID < results[j].CaseID
		}
		if results[i].Model != results[j].Model {
			return results[i].Model < results[j].Model
		}
		return results[i].Repeat < results[j].Repeat
	})
	summary, byModel, byTag := summarize(results)
	return RunResult{
		SuiteName:    spec.Suite.Name,
		SuiteVersion: spec.Suite.Version,
		StartedAt:    startedAt,
		EndedAt:      time.Now().UTC(),
		Results:      results,
		Summary:      summary,
		ByModel:      byModel,
		ByTag:        byTag,
	}, nil
}
