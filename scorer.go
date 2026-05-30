// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/wintermi/sigma"
)

// ScoreInput is the information available to deterministic scorers and judges.
type ScoreInput struct {
	Suite   Suite                  `json:"suite"`
	Case    Case                   `json:"case"`
	Model   sigma.Model            `json:"model"`
	Repeat  int                    `json:"repeat"`
	Request sigma.Request          `json:"request"`
	Output  string                 `json:"output"`
	Message sigma.AssistantMessage `json:"message"`
}

// Scorer evaluates a completed model output.
type Scorer interface {
	Name() string
	Score(context.Context, ScoreInput) (Score, error)
}

// ScorerFunc adapts a function into a Scorer.
type ScorerFunc struct {
	ScorerName string
	Func       func(context.Context, ScoreInput) (Score, error)
}

// Name returns the scorer name.
func (f ScorerFunc) Name() string { return f.ScorerName }

// Score calls f.Func.
func (f ScorerFunc) Score(ctx context.Context, input ScoreInput) (Score, error) {
	if f.Func == nil {
		return Score{}, fmt.Errorf("scorer %q has no function", f.ScorerName)
	}
	return f.Func(ctx, input)
}

// AnswerMatchMode controls how AnswerScorer compares expected answers.
type AnswerMatchMode string

const (
	// MatchExact compares trimmed output with trimmed expected answers.
	MatchExact AnswerMatchMode = "exact"
	// MatchNormalized compares normalized output with normalized expected answers.
	MatchNormalized AnswerMatchMode = "normalized"
	// MatchContains passes when normalized output contains a normalized expected answer.
	MatchContains AnswerMatchMode = "contains"
)

// AutoScorer chooses a deterministic scorer from the case's Expected shape.
type AutoScorer struct{}

// Name implements Scorer.
func (AutoScorer) Name() string { return "auto" }

// Score implements Scorer.
func (AutoScorer) Score(ctx context.Context, input ScoreInput) (Score, error) {
	expected := input.Case.Expected
	switch {
	case len(expected.ToolCalls) > 0:
		return ToolCallScorer{}.Score(ctx, input)
	case len(expected.Choices) > 0:
		return MultipleChoiceScorer{}.Score(ctx, input)
	case expected.JSON != nil:
		return JSONMatchScorer{}.Score(ctx, input)
	case len(expected.Patterns) > 0:
		return RegexScorer{}.Score(ctx, input)
	default:
		return AnswerScorer{Mode: MatchContains}.Score(ctx, input)
	}
}

// AnswerScorer compares output against Expected.Output or Expected.Answers. It
// checks Expected.NegativeAnswers first and fails immediately on a match.
type AnswerScorer struct {
	Mode AnswerMatchMode
}

// Name implements Scorer.
func (s AnswerScorer) Name() string {
	mode := s.Mode
	if mode == "" {
		mode = MatchNormalized
	}
	return "answer_" + string(mode)
}

// Score implements Scorer.
func (s AnswerScorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	mode := s.Mode
	if mode == "" {
		mode = MatchNormalized
	}

	for _, negative := range input.Case.Expected.NegativeAnswers {
		if answerMatches(input.Output, negative, MatchContains) {
			return Score{
				Name:      s.Name(),
				Score:     0,
				Passed:    false,
				Rationale: fmt.Sprintf("output matched negative answer %q", negative),
			}, nil
		}
	}

	answers := expectedAnswers(input.Case.Expected)
	if len(answers) == 0 {
		return Score{}, fmt.Errorf("case %q has no expected answers", input.Case.ID)
	}
	for _, answer := range answers {
		if answerMatches(input.Output, answer, mode) {
			return Score{
				Name:      s.Name(),
				Score:     1,
				Passed:    true,
				Rationale: fmt.Sprintf("output matched expected answer %q", answer),
			}, nil
		}
	}
	return Score{
		Name:      s.Name(),
		Score:     0,
		Passed:    false,
		Rationale: "output did not match any expected answer",
	}, nil
}

// JSONMatchScorer compares output JSON with Expected.JSON, or with the first
// JSON-parsable expected answer.
type JSONMatchScorer struct{}

// Name implements Scorer.
func (JSONMatchScorer) Name() string { return "json_match" }

// Score implements Scorer.
func (JSONMatchScorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	expected, err := expectedJSONValue(input.Case.Expected)
	if err != nil {
		return Score{}, err
	}

	var actual any
	if err := json.Unmarshal([]byte(input.Output), &actual); err != nil {
		return Score{
			Name:      "json_match",
			Score:     0,
			Passed:    false,
			Rationale: "output is not valid JSON",
		}, nil
	}
	actual, err = normalizeJSONValue(actual)
	if err != nil {
		return Score{}, err
	}

	passed := reflect.DeepEqual(actual, expected)
	rationale := "output JSON did not match expected JSON"
	if passed {
		rationale = "output JSON matched expected JSON"
	}
	return Score{Name: "json_match", Score: boolScore(passed), Passed: passed, Rationale: rationale}, nil
}

// RegexScorer passes when any Expected.Patterns regular expression matches.
type RegexScorer struct{}

// Name implements Scorer.
func (RegexScorer) Name() string { return "regex" }

// Score implements Scorer.
func (RegexScorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	if len(input.Case.Expected.Patterns) == 0 {
		return Score{}, fmt.Errorf("case %q has no expected patterns", input.Case.ID)
	}
	for _, pattern := range input.Case.Expected.Patterns {
		matched, err := regexp.MatchString(pattern, input.Output)
		if err != nil {
			return Score{}, fmt.Errorf("compile pattern %q: %w", pattern, err)
		}
		if matched {
			return Score{Name: "regex", Score: 1, Passed: true, Rationale: fmt.Sprintf("output matched pattern %q", pattern)}, nil
		}
	}
	return Score{Name: "regex", Score: 0, Passed: false, Rationale: "output did not match any pattern"}, nil
}

// TokenF1Scorer computes the maximum normalized token F1 over expected answers.
type TokenF1Scorer struct{}

// Name implements Scorer.
func (TokenF1Scorer) Name() string { return "token_f1" }

// Score implements Scorer.
func (TokenF1Scorer) Score(_ context.Context, input ScoreInput) (Score, error) {
	answers := expectedAnswers(input.Case.Expected)
	if len(answers) == 0 {
		return Score{}, fmt.Errorf("case %q has no expected answers", input.Case.ID)
	}
	best := 0.0
	for _, answer := range answers {
		if f1 := tokenF1(input.Output, answer); f1 > best {
			best = f1
		}
	}
	return Score{Name: "token_f1", Score: best, Passed: best >= 1, Rationale: fmt.Sprintf("best token F1 %.3f", best)}, nil
}

func scorersOrDefault(scorers []Scorer) []Scorer {
	if len(scorers) == 0 {
		return []Scorer{AutoScorer{}}
	}
	return scorers
}

func expectedAnswers(expected Expected) []string {
	answers := make([]string, 0, 1+len(expected.Answers))
	if strings.TrimSpace(expected.Output) != "" {
		answers = append(answers, expected.Output)
	}
	answers = append(answers, expected.Answers...)
	return answers
}

func answerMatches(output string, answer string, mode AnswerMatchMode) bool {
	switch mode {
	case MatchExact:
		return strings.TrimSpace(output) == strings.TrimSpace(answer)
	case MatchContains:
		return strings.Contains(NormalizeAnswer(output), NormalizeAnswer(answer))
	default:
		return NormalizeAnswer(output) == NormalizeAnswer(answer)
	}
}

func expectedJSONValue(expected Expected) (any, error) {
	if expected.JSON != nil {
		return normalizeJSONValue(expected.JSON)
	}
	for _, answer := range expectedAnswers(expected) {
		var value any
		if err := json.Unmarshal([]byte(answer), &value); err == nil {
			return normalizeJSONValue(value)
		}
	}
	return nil, fmt.Errorf("expected JSON or JSON answer is required")
}

func normalizeJSONValue(value any) (any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var normalized any
	if err := json.Unmarshal(data, &normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

func boolScore(value bool) float64 {
	if value {
		return 1
	}
	return 0
}
