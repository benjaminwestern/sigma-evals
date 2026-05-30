// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/wintermi/sigma"
)

// RubricGEvalScorer asks a judge for discrete rubric scores and computes the
// final scores from score-token logprobs. It is the multi-metric version of
// ModeGEval.
type RubricGEvalScorer struct {
	Client          Completer
	TargetCompleter TargetCompleter
	Judge           Target
	JudgeModel      sigma.Model
	Rubric          Rubric
	TopLogprobs     int
	JudgeOptions    []sigma.Option
}

// Name implements Scorer.
func (s RubricGEvalScorer) Name() string { return "rubric_g_eval" }

// Score implements Scorer.
func (s RubricGEvalScorer) Score(ctx context.Context, input ScoreInput) (Score, error) {
	rubric := s.Rubric.WithCase(input.Case)
	topLogprobs := s.TopLogprobs
	if topLogprobs <= 0 {
		topLogprobs = defaultTopLogprobs
	}
	judgeTarget := targetWithModelFallback(s.Judge, s.JudgeModel)
	judgeModel := judgeTarget.modelForScoring()
	evaluator := &Evaluator{Client: s.Client, TargetCompleter: s.TargetCompleter}
	judgeResult, err := evaluator.completeTarget(ctx, judgeTarget, sigma.Request{
		Messages: []sigma.Message{sigma.UserText(rubric.Prompt(input))},
	}, appendOptions(
		s.JudgeOptions,
		sigma.WithReasoningLevel(sigma.ThinkingLevelOff),
		withStructuredOutput(judgeModel, map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "rubric_scores",
				"strict": true,
				"schema": rubric.JSONSchema(false),
			},
		}),
		withOpenAILogprobs(topLogprobs),
	), map[string]any{"role": "judge", "mode": "rubric_g_eval"})
	if err != nil {
		return Score{}, err
	}
	rawOutput, err := targetResultText(judgeResult)
	if err != nil {
		return Score{}, ErrGEvalLogprobsRequired
	}
	logprobs, ok := targetResultLogprobs(judgeResult)
	if !ok {
		return Score{}, ErrGEvalLogprobsRequired
	}
	scores, err := RubricGEvalScoreForOutput(rubric, rawOutput, logprobs)
	if err != nil {
		return Score{}, err
	}
	threshold := rubric.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}
	return Score{
		Name:      s.Name(),
		Score:     scores.Aggregate,
		Passed:    scores.Aggregate >= threshold,
		Rationale: "G-Eval logprob-weighted rubric aggregate",
		Details: map[string]any{
			"rawJudgeOutput": rawOutput,
			"rubricScores":   scores,
			"logprobs":       logprobs,
		},
	}, nil
}

// RubricGEvalScoreForOutput computes logprob-weighted rubric scores from a
// structured judge output and its token logprobs.
func RubricGEvalScoreForOutput(rubric Rubric, rawOutput string, logprobs []TokenLogprob) (RubricScores, error) {
	rubric = rubric.WithCase(Case{})
	if len(logprobs) == 0 {
		return RubricScores{}, ErrGEvalLogprobsRequired
	}
	rawFromLogprobs := rawOutputFromLogprobs(logprobs)
	if strings.TrimSpace(rawFromLogprobs) != "" {
		rawOutput = rawFromLogprobs
	}
	offsets := rubricMetricOffsets(rawOutput, rubric)
	if len(offsets) == 0 {
		return RubricScores{}, fmt.Errorf("%w: no rubric metric keys found in judge output", ErrGEvalLogprobsRequired)
	}

	raw := make(map[string]any, len(rubric.Dimensions))
	scores := make(map[string]float64, len(rubric.Dimensions))
	normalized := make(map[string]float64, len(rubric.Dimensions))
	var totalWeight float64
	var weighted float64
	for _, dimension := range rubric.Dimensions {
		key := dimension.JSONKey()
		start, ok := offsets[key]
		if !ok {
			return RubricScores{}, fmt.Errorf("%w: missing logprob range for metric %q", ErrGEvalLogprobsRequired, key)
		}
		end := nextMetricOffset(start, offsets, len(rawOutput))
		rawScore, normScore, ok := rubricScoreInRange(logprobs, start, end, dimension.Type)
		if !ok {
			return RubricScores{}, fmt.Errorf("%w: no score token logprobs for metric %q", ErrGEvalLogprobsRequired, key)
		}
		weight := dimension.Weight
		if weight <= 0 {
			weight = 1
		}
		totalWeight += weight
		weighted += normScore * weight
		raw[key] = rawScore
		scores[key] = rawScore
		normalized[key] = normScore
	}
	if totalWeight == 0 {
		return RubricScores{}, fmt.Errorf("rubric has no positive weights")
	}
	return RubricScores{Raw: raw, Scores: scores, Normalized: normalized, Aggregate: weighted / totalWeight}, nil
}

func rawOutputFromLogprobs(logprobs []TokenLogprob) string {
	var b strings.Builder
	for _, logprob := range logprobs {
		b.WriteString(logprob.Token)
	}
	return b.String()
}

func rubricMetricOffsets(rawOutput string, rubric Rubric) map[string]int {
	type metricOffset struct {
		key    string
		offset int
	}
	found := make([]metricOffset, 0, len(rubric.Dimensions))
	for _, dimension := range rubric.Dimensions {
		key := dimension.JSONKey()
		offset := strings.Index(rawOutput, `"`+key+`"`)
		if offset < 0 {
			offset = strings.Index(rawOutput, key)
		}
		if offset >= 0 {
			found = append(found, metricOffset{key: key, offset: offset + len(key)})
		}
	}
	sort.Slice(found, func(i, j int) bool { return found[i].offset < found[j].offset })
	out := make(map[string]int, len(found))
	for _, item := range found {
		out[item.key] = item.offset
	}
	return out
}

func nextMetricOffset(start int, offsets map[string]int, fallback int) int {
	end := fallback
	for _, offset := range offsets {
		if offset > start && offset < end {
			end = offset
		}
	}
	return end
}

func rubricScoreInRange(logprobs []TokenLogprob, start int, end int, ratingType RatingType) (float64, float64, bool) {
	offset := 0
	for _, logprob := range logprobs {
		next := offset + len(logprob.Token)
		if next > start && offset < end {
			if raw, normalized, ok := weightedRubricTokenScore(logprob, ratingType); ok {
				return raw, normalized, true
			}
		}
		offset = next
		if offset >= end {
			break
		}
	}
	return 0, 0, false
}

func weightedRubricTokenScore(tokenLogprob TokenLogprob, ratingType RatingType) (float64, float64, bool) {
	primaryRaw, primaryNorm, ok := rubricTokenScore(tokenLogprob.Token, ratingType)
	if !ok {
		return 0, 0, false
	}
	if len(tokenLogprob.TopLogprobs) == 0 {
		return primaryRaw, primaryNorm, true
	}

	var totalProb float64
	var rawWeighted float64
	var normWeighted float64
	containsPrimary := false
	for _, top := range tokenLogprob.TopLogprobs {
		if top.Token == tokenLogprob.Token {
			containsPrimary = true
		}
		raw, norm, ok := rubricTokenScore(top.Token, ratingType)
		if !ok {
			continue
		}
		probability := math.Exp(top.Logprob)
		if probability <= 0 || math.IsInf(probability, 0) || math.IsNaN(probability) {
			continue
		}
		totalProb += probability
		rawWeighted += raw * probability
		normWeighted += norm * probability
	}
	if !containsPrimary {
		probability := math.Exp(tokenLogprob.Logprob)
		if tokenLogprob.Logprob == -9999 || probability <= 0 || math.IsInf(probability, 0) || math.IsNaN(probability) {
			probability = 1
		}
		totalProb += probability
		rawWeighted += primaryRaw * probability
		normWeighted += primaryNorm * probability
	}
	if totalProb == 0 {
		return 0, 0, false
	}
	return rawWeighted / totalProb, normWeighted / totalProb, true
}

func rubricTokenScore(token string, ratingType RatingType) (float64, float64, bool) {
	cleaned := strings.Trim(strings.ToLower(strings.TrimSpace(token)), `"',:`)
	switch ratingType {
	case RatingPassFail:
		switch cleaned {
		case "pass":
			return 1, 1, true
		case "fail":
			return 0, 0, true
		}
	case RatingPassFailCritical:
		switch cleaned {
		case "pass":
			return 1, 1, true
		case "fail":
			return 0, 0.5, true
		case "critical":
			return -1, 0, true
		}
	default:
		score, ok := scoreToken(cleaned)
		if ok {
			return score, (score - 1) / 4, true
		}
	}
	return 0, 0, false
}
