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
	"unicode"

	"github.com/wintermi/sigma"
)

// RatingType identifies a rubric score domain.
type RatingType string

const (
	// RatingFiveStar asks for an integer 1-5 rating.
	RatingFiveStar RatingType = "five_star"
	// RatingPassFail asks for pass/fail.
	RatingPassFail RatingType = "pass_fail"
	// RatingPassFailCritical asks for pass/fail/critical, where critical is a severe failure.
	RatingPassFailCritical RatingType = "pass_fail_critical"
)

// Rubric is a multi-dimensional judge rubric. Dimensions are converted into a
// strict JSON schema for judges, then normalized to a 0-1 aggregate score.
type Rubric struct {
	Name        string            `json:"name,omitempty"`
	Description string            `json:"description,omitempty"`
	Dimensions  []RubricDimension `json:"dimensions"`
	Threshold   float64           `json:"threshold,omitempty"`
}

// RubricDimension is one named score the judge must return.
type RubricDimension struct {
	Name        string     `json:"name"`
	Instruction string     `json:"instruction,omitempty"`
	Type        RatingType `json:"type"`
	Weight      float64    `json:"weight,omitempty"`
}

// RubricScores contains parsed raw and normalized dimension scores.
type RubricScores struct {
	Raw        map[string]any     `json:"raw"`
	Scores     map[string]float64 `json:"scores"`
	Normalized map[string]float64 `json:"normalized"`
	Aggregate  float64            `json:"aggregate"`
}

// WithCase returns a copy of the rubric adapted with case-specific rubric text.
func (r Rubric) WithCase(c Case) Rubric {
	out := r
	out.Dimensions = append([]RubricDimension(nil), r.Dimensions...)
	if len(out.Dimensions) == 0 {
		out.Dimensions = []RubricDimension{{Name: "Overall", Type: RatingFiveStar, Instruction: "Rate the target output against the expected answer and rubric."}}
	}
	if strings.TrimSpace(c.Expected.Rubric) != "" {
		if strings.TrimSpace(out.Description) != "" {
			out.Description += "\n\n"
		}
		out.Description += "Case-specific rubric:\n" + c.Expected.Rubric
	}
	return out
}

// JSONSchema builds the strict judge output schema. When allowFloat is false,
// judges are forced to discrete tokens; when true, final normalized outputs can
// use floats.
func (r Rubric) JSONSchema(allowFloat bool) map[string]any {
	properties := map[string]any{}
	required := make([]string, 0, len(r.Dimensions))
	for _, dimension := range r.Dimensions {
		key := dimension.JSONKey()
		required = append(required, key)
		properties[key] = dimensionSchema(dimension, allowFloat)
	}
	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

// Prompt renders the rubric against one output to judge.
func (r Rubric) Prompt(input ScoreInput) string {
	r = r.WithCase(input.Case)
	var b strings.Builder
	if strings.TrimSpace(r.Name) != "" {
		b.WriteString("Evaluation: ")
		b.WriteString(r.Name)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(r.Description) != "" {
		b.WriteString(r.Description)
		b.WriteString("\n\n")
	}
	b.WriteString("Return a JSON object with exactly these rating fields:\n")
	for _, dimension := range r.Dimensions {
		b.WriteString("- ")
		b.WriteString(dimension.JSONKey())
		b.WriteString(": ")
		b.WriteString(dimension.Instruction)
		if dimension.Instruction != "" {
			b.WriteString(" ")
		}
		b.WriteString(ratingInstruction(dimension.Type))
		b.WriteString("\n")
	}
	b.WriteString("\n")
	if dataType := caseDataType(input.Suite, input.Case); dataType != "" {
		b.WriteString("Evaluation data type: ")
		b.WriteString(string(dataType))
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(input.Case.Input) != "" {
		b.WriteString("Input:\n")
		b.WriteString(input.Case.Input)
		b.WriteString("\n\n")
	}
	if traceText := formatTrace(input.Case.Trace); traceText != "" {
		b.WriteString("Task trace:\n")
		b.WriteString(traceText)
		b.WriteString("\n\n")
	}
	if groundTruth := strings.Join(expectedAnswers(input.Case.Expected), "\n"); strings.TrimSpace(groundTruth) != "" {
		b.WriteString("Expected answer/reference:\n")
		b.WriteString(groundTruth)
		b.WriteString("\n\n")
	}
	b.WriteString("Target output:\n")
	b.WriteString(input.Output)
	b.WriteString("\n")
	return b.String()
}

// ParseScores parses and normalizes judge output according to the rubric.
func (r Rubric) ParseScores(text string) (RubricScores, error) {
	r = r.WithCase(Case{})
	var raw map[string]any
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return RubricScores{}, err
	}
	return r.ScoreRaw(raw)
}

// ScoreRaw normalizes raw judge fields and computes a weighted aggregate.
func (r Rubric) ScoreRaw(raw map[string]any) (RubricScores, error) {
	r = r.WithCase(Case{})
	scores := make(map[string]float64, len(r.Dimensions))
	normalized := make(map[string]float64, len(r.Dimensions))
	var totalWeight float64
	var weighted float64
	for _, dimension := range r.Dimensions {
		key := dimension.JSONKey()
		value, ok := raw[key]
		if !ok {
			return RubricScores{}, fmt.Errorf("missing rubric score %q", key)
		}
		rawScore, normScore, err := scoreRubricValue(value, dimension.Type)
		if err != nil {
			return RubricScores{}, fmt.Errorf("score %q: %w", key, err)
		}
		weight := dimension.Weight
		if weight <= 0 {
			weight = 1
		}
		totalWeight += weight
		weighted += normScore * weight
		scores[key] = rawScore
		normalized[key] = normScore
	}
	if totalWeight == 0 {
		return RubricScores{}, fmt.Errorf("rubric has no positive weights")
	}
	return RubricScores{Raw: raw, Scores: scores, Normalized: normalized, Aggregate: weighted / totalWeight}, nil
}

// JSONKey returns a stable JSON key for the dimension.
func (d RubricDimension) JSONKey() string {
	return jsonKey(d.Name)
}

// RubricJudgeScorer asks an LLM judge for a multi-dimensional rubric result.
type RubricJudgeScorer struct {
	Client       Completer
	JudgeModel   sigma.Model
	Rubric       Rubric
	JudgeOptions []sigma.Option
}

// Name implements Scorer.
func (s RubricJudgeScorer) Name() string { return "rubric_judge" }

// Score implements Scorer.
func (s RubricJudgeScorer) Score(ctx context.Context, input ScoreInput) (Score, error) {
	rubric := s.Rubric.WithCase(input.Case)
	final, err := clientOrDefault(s.Client).Complete(ctx, s.JudgeModel, sigma.Request{
		Messages: []sigma.Message{sigma.UserText(rubric.Prompt(input))},
	}, appendOptions(s.JudgeOptions, withStructuredOutput(s.JudgeModel, map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "rubric_scores",
			"strict": true,
			"schema": rubric.JSONSchema(false),
		},
	}))...)
	if err != nil {
		return Score{}, err
	}
	rawOutput, err := AssistantText(final)
	if err != nil {
		return Score{}, err
	}
	rubricScores, err := rubric.ParseScores(rawOutput)
	if err != nil {
		return Score{}, err
	}
	threshold := rubric.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	return Score{
		Name:      s.Name(),
		Score:     rubricScores.Aggregate,
		Passed:    rubricScores.Aggregate >= threshold,
		Rationale: "weighted rubric aggregate",
		Details: map[string]any{
			"rawJudgeOutput": rawOutput,
			"rubricScores":   rubricScores,
		},
	}, nil
}

func clientOrDefault(client Completer) Completer {
	if client == nil {
		return sigma.NewClient()
	}
	return client
}

func dimensionSchema(dimension RubricDimension, allowFloat bool) map[string]any {
	schema := map[string]any{
		"title":       dimension.Name,
		"description": strings.TrimSpace(dimension.Instruction + "\n\n" + ratingInstruction(dimension.Type)),
	}
	switch dimension.Type {
	case RatingPassFail:
		if allowFloat {
			schema["type"] = "number"
			schema["minimum"] = 0
			schema["maximum"] = 1
		} else {
			schema["type"] = "string"
			schema["enum"] = []string{"pass", "fail"}
		}
	case RatingPassFailCritical:
		if allowFloat {
			schema["type"] = "number"
			schema["minimum"] = -1
			schema["maximum"] = 1
		} else {
			schema["type"] = "string"
			schema["enum"] = []string{"pass", "fail", "critical"}
		}
	default:
		if allowFloat {
			schema["type"] = "number"
		} else {
			schema["type"] = "integer"
		}
		schema["minimum"] = 1
		schema["maximum"] = 5
	}
	return schema
}

func ratingInstruction(ratingType RatingType) string {
	switch ratingType {
	case RatingPassFail:
		return "The rating must be either 'pass' or 'fail'."
	case RatingPassFailCritical:
		return "The rating must be 'pass', 'fail', or 'critical', where critical means a severe failure."
	default:
		return "The rating must be an integer from 1 to 5, where 1 is worst and 5 is best."
	}
}

func scoreRubricValue(value any, ratingType RatingType) (float64, float64, error) {
	switch ratingType {
	case RatingPassFail:
		score, err := categoricalScore(value, map[string]float64{"fail": 0, "pass": 1})
		return score, score, err
	case RatingPassFailCritical:
		score, err := categoricalScore(value, map[string]float64{"critical": -1, "fail": 0, "pass": 1})
		return score, (score + 1) / 2, err
	default:
		score, ok := numberValue(value)
		if !ok || score < 1 || score > 5 {
			return 0, 0, fmt.Errorf("expected a 1-5 rating")
		}
		return score, (score - 1) / 4, nil
	}
}

func categoricalScore(value any, values map[string]float64) (float64, error) {
	if score, ok := numberValue(value); ok {
		return score, nil
	}
	text, ok := value.(string)
	if !ok {
		return 0, fmt.Errorf("expected categorical score")
	}
	score, ok := values[strings.ToLower(strings.TrimSpace(text))]
	if !ok {
		return 0, fmt.Errorf("unknown categorical score %q", text)
	}
	return score, nil
}

func jsonKey(name string) string {
	var b strings.Builder
	lastUnderscore := false
	for _, r := range strings.ToLower(name) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore && b.Len() > 0 {
			b.WriteRune('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(b.String(), "_")
}

func caseDataType(suite Suite, c Case) EvalDataType {
	if c.DataType != "" {
		return c.DataType
	}
	return suite.DataType
}

func formatTrace(trace Trace) string {
	if len(trace.Messages) == 0 && len(trace.Events) == 0 {
		return ""
	}
	data, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		return fmt.Sprint(trace)
	}
	return string(data)
}
