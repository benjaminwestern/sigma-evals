// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"strings"
)

// MultipleChoiceScorer extracts a selected option and compares it to Expected.CorrectChoices.
type MultipleChoiceScorer struct{}

// Name implements Scorer.
func (MultipleChoiceScorer) Name() string { return "multiple_choice" }

// Score implements Scorer.
func (MultipleChoiceScorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	if len(input.Case.Expected.Choices) == 0 {
		return Score{}, fmt.Errorf("case %q has no choices", input.Case.ID)
	}
	correct := input.Case.Expected.CorrectChoices
	if len(correct) == 0 {
		correct = input.Case.Expected.Answers
	}
	if len(correct) == 0 {
		return Score{}, fmt.Errorf("case %q has no correct choices", input.Case.ID)
	}

	picked, ok := PickChoice(input.Output, input.Case.Expected.Choices)
	if !ok {
		return Score{Name: "multiple_choice", Score: 0, Passed: false, Rationale: "no choice could be extracted"}, nil
	}
	passed := containsChoice(correct, picked)
	rationale := fmt.Sprintf("picked %q", picked)
	if !passed {
		rationale = fmt.Sprintf("picked %q, expected one of %v", picked, correct)
	}
	return Score{
		Name:      "multiple_choice",
		Score:     boolScore(passed),
		Passed:    passed,
		Rationale: rationale,
		Details:   map[string]any{"picked": picked, "correct": correct},
	}, nil
}

// PickChoice extracts the first matching choice label or choice text from output.
func PickChoice(output string, choices []Choice) (string, bool) {
	labelFields := strings.FieldsFunc(strings.ToLower(output), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
	for _, choice := range choices {
		label := strings.ToLower(strings.TrimSpace(choice.Label))
		if label == "" {
			continue
		}
		for _, field := range labelFields {
			if field == label {
				return choice.Label, true
			}
		}
	}

	normalizedOutput := NormalizeAnswer(output)
	for _, choice := range choices {
		text := NormalizeAnswer(choice.Text)
		if text != "" && strings.Contains(normalizedOutput, text) {
			return choice.Label, true
		}
	}
	return "", false
}

func containsChoice(choices []string, picked string) bool {
	picked = NormalizeAnswer(picked)
	for _, choice := range choices {
		if NormalizeAnswer(choice) == picked {
			return true
		}
	}
	return false
}
