// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"sort"
	"strings"
	"time"
)

const (
	// VarianceLayerSuite identifies normal suite/scorer result samples.
	VarianceLayerSuite = "suite"
	// VarianceLayerBatchJudge identifies batch judge result samples.
	VarianceLayerBatchJudge = "batch_judge"
	// VarianceLayerJudgeAlignment identifies judge-alignment result samples.
	VarianceLayerJudgeAlignment = "judge_alignment"
)

// VarianceSample is one comparable observation from any eval layer.
type VarianceSample struct {
	Layer         string   `json:"layer"`
	RunName       string   `json:"runName,omitempty"`
	CaseID        string   `json:"caseId"`
	CaseName      string   `json:"caseName,omitempty"`
	Model         string   `json:"model,omitempty"`
	Provider      string   `json:"provider,omitempty"`
	Scorer        string   `json:"scorer,omitempty"`
	Repeat        int      `json:"repeat,omitempty"`
	Score         float64  `json:"score,omitempty"`
	Passed        bool     `json:"passed"`
	Error         string   `json:"error,omitempty"`
	DurationMS    int64    `json:"durationMs,omitempty"`
	Output        string   `json:"output,omitempty"`
	OutputHash    string   `json:"outputHash,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	ExpectedScore float64  `json:"expectedScore,omitempty"`
	ScoreError    float64  `json:"scoreError,omitempty"`
}

// VarianceGroupKey identifies the sample grouping used for stability and delta
// reports. The default grouping is layer + case + model + scorer.
type VarianceGroupKey struct {
	Layer  string `json:"layer"`
	CaseID string `json:"caseId"`
	Model  string `json:"model,omitempty"`
	Scorer string `json:"scorer,omitempty"`
}

// VarianceReport summarizes repeated eval samples as distributions.
type VarianceReport struct {
	Name        string               `json:"name,omitempty"`
	GeneratedAt time.Time            `json:"generatedAt"`
	Total       VarianceStats        `json:"total"`
	Groups      []VarianceGroupStats `json:"groups"`
}

// VarianceGroupStats summarizes repeated samples for one comparable group.
type VarianceGroupStats struct {
	Key              VarianceGroupKey `json:"key"`
	Count            int              `json:"count"`
	Errors           int              `json:"errors"`
	Passed           int              `json:"passed"`
	Failed           int              `json:"failed"`
	PassRate         float64          `json:"passRate,omitempty"`
	MeanScore        float64          `json:"meanScore,omitempty"`
	StdDevScore      float64          `json:"stdDevScore,omitempty"`
	StdErrScore      float64          `json:"stdErrScore,omitempty"`
	MinScore         float64          `json:"minScore,omitempty"`
	MaxScore         float64          `json:"maxScore,omitempty"`
	MeanDurationMS   float64          `json:"meanDurationMs,omitempty"`
	UniqueOutputs    int              `json:"uniqueOutputs,omitempty"`
	OutputDiversity  float64          `json:"outputDiversity,omitempty"`
	MeanScoreError   float64          `json:"meanScoreError,omitempty"`
	StdDevScoreError float64          `json:"stdDevScoreError,omitempty"`
}

// VarianceStats summarizes the whole sample set.
type VarianceStats struct {
	Count           int     `json:"count"`
	Errors          int     `json:"errors"`
	Passed          int     `json:"passed"`
	Failed          int     `json:"failed"`
	PassRate        float64 `json:"passRate,omitempty"`
	MeanScore       float64 `json:"meanScore,omitempty"`
	StdDevScore     float64 `json:"stdDevScore,omitempty"`
	StdErrScore     float64 `json:"stdErrScore,omitempty"`
	UniqueOutputs   int     `json:"uniqueOutputs,omitempty"`
	OutputDiversity float64 `json:"outputDiversity,omitempty"`
}

// VarianceCompareOptions configures baseline/current distribution comparison.
type VarianceCompareOptions struct {
	// ConfidenceZ is the two-sided z threshold. Defaults to 1.96.
	ConfidenceZ float64 `json:"confidenceZ,omitempty"`
	// MinPassRateDelta ignores smaller absolute pass-rate changes when deciding direction.
	MinPassRateDelta float64 `json:"minPassRateDelta,omitempty"`
	// MinMeanScoreDelta ignores smaller absolute mean-score changes when deciding direction.
	MinMeanScoreDelta float64 `json:"minMeanScoreDelta,omitempty"`
}

// VarianceComparison compares a current run distribution against a baseline.
type VarianceComparison struct {
	BaselineName string                    `json:"baselineName,omitempty"`
	CurrentName  string                    `json:"currentName,omitempty"`
	Options      VarianceCompareOptions    `json:"options"`
	Summary      VarianceComparisonSummary `json:"summary"`
	Deltas       []VarianceDelta           `json:"deltas"`
}

// VarianceComparisonSummary summarizes delta directions.
type VarianceComparisonSummary struct {
	Groups       int `json:"groups"`
	Regressions  int `json:"regressions"`
	Improvements int `json:"improvements"`
	Stable       int `json:"stable"`
	Missing      int `json:"missing"`
	New          int `json:"new"`
}

// VarianceDelta compares one baseline/current distribution group.
type VarianceDelta struct {
	Key                 VarianceGroupKey    `json:"key"`
	Baseline            *VarianceGroupStats `json:"baseline,omitempty"`
	Current             *VarianceGroupStats `json:"current,omitempty"`
	DeltaPassRate       float64             `json:"deltaPassRate,omitempty"`
	DeltaMeanScore      float64             `json:"deltaMeanScore,omitempty"`
	DeltaStdDevScore    float64             `json:"deltaStdDevScore,omitempty"`
	DeltaMeanDurationMS float64             `json:"deltaMeanDurationMs,omitempty"`
	ScoreZ              float64             `json:"scoreZ,omitempty"`
	PassRateZ           float64             `json:"passRateZ,omitempty"`
	EffectSize          float64             `json:"effectSize,omitempty"`
	Significant         bool                `json:"significant"`
	Direction           string              `json:"direction"`
}

// VarianceSamplesFromRunResult extracts comparable samples from a normal suite run.
func VarianceSamplesFromRunResult(run RunResult) []VarianceSample {
	out := make([]VarianceSample, 0, len(run.Results))
	for _, result := range run.Results {
		if len(result.Scores) == 0 {
			out = append(out, VarianceSample{
				Layer:      VarianceLayerSuite,
				RunName:    run.SuiteName,
				CaseID:     result.CaseID,
				CaseName:   result.CaseName,
				Model:      result.Model,
				Provider:   string(result.Provider),
				Repeat:     result.Repeat,
				Passed:     result.Error == "",
				Error:      result.Error,
				DurationMS: result.DurationMS,
				Output:     result.Output,
				OutputHash: outputHash(result.Output),
				Tags:       append([]string(nil), result.Tags...),
			})
			continue
		}
		for _, score := range result.Scores {
			out = append(out, VarianceSample{
				Layer:      VarianceLayerSuite,
				RunName:    run.SuiteName,
				CaseID:     result.CaseID,
				CaseName:   result.CaseName,
				Model:      result.Model,
				Provider:   string(result.Provider),
				Scorer:     score.Name,
				Repeat:     result.Repeat,
				Score:      score.Score,
				Passed:     score.Passed,
				Error:      result.Error,
				DurationMS: result.DurationMS,
				Output:     result.Output,
				OutputHash: outputHash(result.Output),
				Tags:       append([]string(nil), result.Tags...),
			})
		}
	}
	return out
}

// VarianceSamplesFromBatchJudgeResult extracts comparable samples from a batch judge run.
func VarianceSamplesFromBatchJudgeResult(run BatchJudgeResult) []VarianceSample {
	out := make([]VarianceSample, 0, len(run.Results))
	for _, result := range run.Results {
		out = append(out, VarianceSample{
			Layer:      VarianceLayerBatchJudge,
			RunName:    run.Name,
			CaseID:     result.CaseID,
			CaseName:   result.CaseName,
			Model:      result.Target,
			Provider:   string(result.TargetProvider),
			Scorer:     firstNonEmpty(string(result.Mode), "judge"),
			Repeat:     result.Repeat,
			Score:      result.Score,
			Passed:     result.Passed,
			Error:      result.Error,
			DurationMS: result.DurationMS,
			Output:     result.TargetOutput,
			OutputHash: outputHash(result.TargetOutput),
			Tags:       append([]string(nil), result.Tags...),
		})
	}
	return out
}

// VarianceSamplesFromJudgeAlignmentResult extracts comparable samples from a judge-alignment run.
func VarianceSamplesFromJudgeAlignmentResult(run JudgeAlignmentRunResult) []VarianceSample {
	out := make([]VarianceSample, 0, len(run.Results))
	for _, result := range run.Results {
		out = append(out, VarianceSample{
			Layer:         VarianceLayerJudgeAlignment,
			RunName:       run.Name,
			CaseID:        result.CaseID,
			Model:         result.Model,
			Provider:      string(result.Provider),
			Scorer:        "judge_alignment",
			Score:         result.ActualScore,
			Passed:        result.PassedMatch,
			Error:         result.Error,
			DurationMS:    result.DurationMS,
			ExpectedScore: result.ExpectedScore,
			ScoreError:    result.ScoreError,
		})
	}
	return out
}

// VarianceReportFromRunResult summarizes stability for a normal suite run.
func VarianceReportFromRunResult(run RunResult) VarianceReport {
	return BuildVarianceReport(run.SuiteName, VarianceSamplesFromRunResult(run))
}

// VarianceReportFromBatchJudgeResult summarizes stability for a batch judge run.
func VarianceReportFromBatchJudgeResult(run BatchJudgeResult) VarianceReport {
	return BuildVarianceReport(run.Name, VarianceSamplesFromBatchJudgeResult(run))
}

// VarianceReportFromJudgeAlignmentResult summarizes stability for judge alignment.
func VarianceReportFromJudgeAlignmentResult(run JudgeAlignmentRunResult) VarianceReport {
	return BuildVarianceReport(run.Name, VarianceSamplesFromJudgeAlignmentResult(run))
}

// BuildVarianceReport summarizes repeated samples across all eval layers.
func BuildVarianceReport(name string, samples []VarianceSample) VarianceReport {
	groups := make(map[VarianceGroupKey][]VarianceSample)
	for _, sample := range samples {
		key := VarianceGroupKey{Layer: sample.Layer, CaseID: sample.CaseID, Model: sample.Model, Scorer: sample.Scorer}
		groups[key] = append(groups[key], sample)
	}
	keys := make([]VarianceGroupKey, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return varianceKeyLess(keys[i], keys[j]) })

	stats := make([]VarianceGroupStats, 0, len(keys))
	var totalSamples []VarianceSample
	for _, key := range keys {
		stats = append(stats, computeVarianceGroupStats(key, groups[key]))
		totalSamples = append(totalSamples, groups[key]...)
	}
	return VarianceReport{
		Name:        name,
		GeneratedAt: time.Now().UTC(),
		Total:       computeVarianceStats(totalSamples),
		Groups:      stats,
	}
}

// CompareVarianceReports compares current distributions against baseline distributions.
func CompareVarianceReports(baseline VarianceReport, current VarianceReport, options VarianceCompareOptions) VarianceComparison {
	if options.ConfidenceZ <= 0 {
		options.ConfidenceZ = 1.96
	}
	baselineByKey := map[VarianceGroupKey]VarianceGroupStats{}
	currentByKey := map[VarianceGroupKey]VarianceGroupStats{}
	keysMap := map[VarianceGroupKey]struct{}{}
	for _, group := range baseline.Groups {
		baselineByKey[group.Key] = group
		keysMap[group.Key] = struct{}{}
	}
	for _, group := range current.Groups {
		currentByKey[group.Key] = group
		keysMap[group.Key] = struct{}{}
	}
	keys := make([]VarianceGroupKey, 0, len(keysMap))
	for key := range keysMap {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return varianceKeyLess(keys[i], keys[j]) })

	comparison := VarianceComparison{BaselineName: baseline.Name, CurrentName: current.Name, Options: options}
	for _, key := range keys {
		base, hasBase := baselineByKey[key]
		curr, hasCurr := currentByKey[key]
		delta := VarianceDelta{Key: key}
		if hasBase {
			delta.Baseline = &base
		}
		if hasCurr {
			delta.Current = &curr
		}
		switch {
		case !hasBase:
			delta.Direction = "new"
			comparison.Summary.New++
		case !hasCurr:
			delta.Direction = "missing"
			comparison.Summary.Missing++
		default:
			delta = compareVarianceGroup(base, curr, options)
			switch delta.Direction {
			case "regression":
				comparison.Summary.Regressions++
			case "improvement":
				comparison.Summary.Improvements++
			default:
				comparison.Summary.Stable++
			}
		}
		comparison.Deltas = append(comparison.Deltas, delta)
	}
	comparison.Summary.Groups = len(comparison.Deltas)
	return comparison
}

func compareVarianceGroup(base VarianceGroupStats, curr VarianceGroupStats, options VarianceCompareOptions) VarianceDelta {
	delta := VarianceDelta{Key: base.Key, Baseline: &base, Current: &curr}
	delta.DeltaPassRate = curr.PassRate - base.PassRate
	delta.DeltaMeanScore = curr.MeanScore - base.MeanScore
	delta.DeltaStdDevScore = curr.StdDevScore - base.StdDevScore
	delta.DeltaMeanDurationMS = curr.MeanDurationMS - base.MeanDurationMS
	delta.ScoreZ = scoreZ(base, curr, delta.DeltaMeanScore)
	delta.PassRateZ = passRateZ(base, curr, delta.DeltaPassRate)
	delta.EffectSize = effectSize(base, curr, delta.DeltaMeanScore)
	delta.Significant = math.Abs(delta.ScoreZ) >= options.ConfidenceZ || math.Abs(delta.PassRateZ) >= options.ConfidenceZ

	meanScoreMoved := math.Abs(delta.DeltaMeanScore) >= options.MinMeanScoreDelta
	passRateMoved := math.Abs(delta.DeltaPassRate) >= options.MinPassRateDelta
	if !delta.Significant || (!meanScoreMoved && !passRateMoved) {
		delta.Direction = "stable"
		return delta
	}
	if delta.DeltaPassRate < -options.MinPassRateDelta || delta.DeltaMeanScore < -options.MinMeanScoreDelta {
		delta.Direction = "regression"
		return delta
	}
	if delta.DeltaPassRate > options.MinPassRateDelta || delta.DeltaMeanScore > options.MinMeanScoreDelta {
		delta.Direction = "improvement"
		return delta
	}
	delta.Direction = "stable"
	return delta
}

func computeVarianceGroupStats(key VarianceGroupKey, samples []VarianceSample) VarianceGroupStats {
	stats := VarianceGroupStats{Key: key}
	var scoreValues []float64
	var scoreErrors []float64
	var durationSum int64
	outputs := make(map[string]struct{})
	for _, sample := range samples {
		stats.Count++
		durationSum += sample.DurationMS
		if sample.OutputHash != "" {
			outputs[sample.OutputHash] = struct{}{}
		}
		if sample.Error != "" {
			stats.Errors++
			continue
		}
		scoreValues = append(scoreValues, sample.Score)
		scoreErrors = append(scoreErrors, sample.ScoreError)
		if sample.Passed {
			stats.Passed++
		} else {
			stats.Failed++
		}
	}
	valid := stats.Passed + stats.Failed
	stats.PassRate = safeDiv(float64(stats.Passed), float64(valid))
	stats.MeanScore = mean(scoreValues)
	stats.StdDevScore = stddev(scoreValues)
	stats.StdErrScore = safeDiv(stats.StdDevScore, math.Sqrt(float64(len(scoreValues))))
	stats.MinScore = minValue(scoreValues)
	stats.MaxScore = maxValue(scoreValues)
	stats.MeanDurationMS = safeDiv(float64(durationSum), float64(stats.Count))
	stats.UniqueOutputs = len(outputs)
	stats.OutputDiversity = safeDiv(float64(stats.UniqueOutputs), float64(valid))
	stats.MeanScoreError = mean(scoreErrors)
	stats.StdDevScoreError = stddev(scoreErrors)
	return stats
}

func computeVarianceStats(samples []VarianceSample) VarianceStats {
	var stats VarianceStats
	var scores []float64
	outputs := make(map[string]struct{})
	for _, sample := range samples {
		stats.Count++
		if sample.OutputHash != "" {
			outputs[sample.OutputHash] = struct{}{}
		}
		if sample.Error != "" {
			stats.Errors++
			continue
		}
		scores = append(scores, sample.Score)
		if sample.Passed {
			stats.Passed++
		} else {
			stats.Failed++
		}
	}
	valid := stats.Passed + stats.Failed
	stats.PassRate = safeDiv(float64(stats.Passed), float64(valid))
	stats.MeanScore = mean(scores)
	stats.StdDevScore = stddev(scores)
	stats.StdErrScore = safeDiv(stats.StdDevScore, math.Sqrt(float64(len(scores))))
	stats.UniqueOutputs = len(outputs)
	stats.OutputDiversity = safeDiv(float64(stats.UniqueOutputs), float64(valid))
	return stats
}

func scoreZ(base VarianceGroupStats, curr VarianceGroupStats, diff float64) float64 {
	baseN := base.Passed + base.Failed
	currN := curr.Passed + curr.Failed
	se := math.Sqrt(safeDiv(base.StdDevScore*base.StdDevScore, float64(baseN)) + safeDiv(curr.StdDevScore*curr.StdDevScore, float64(currN)))
	return zFromSE(diff, se)
}

func passRateZ(base VarianceGroupStats, curr VarianceGroupStats, diff float64) float64 {
	baseN := base.Passed + base.Failed
	currN := curr.Passed + curr.Failed
	if baseN == 0 || currN == 0 {
		return 0
	}
	pooled := safeDiv(float64(base.Passed+curr.Passed), float64(baseN+currN))
	se := math.Sqrt(pooled * (1 - pooled) * (safeDiv(1, float64(baseN)) + safeDiv(1, float64(currN))))
	return zFromSE(diff, se)
}

func effectSize(base VarianceGroupStats, curr VarianceGroupStats, diff float64) float64 {
	baseN := base.Passed + base.Failed
	currN := curr.Passed + curr.Failed
	denominator := float64(baseN + currN - 2)
	if denominator <= 0 {
		return 0
	}
	pooledVariance := safeDiv(float64(baseN-1)*base.StdDevScore*base.StdDevScore+float64(currN-1)*curr.StdDevScore*curr.StdDevScore, denominator)
	return safeDiv(diff, math.Sqrt(pooledVariance))
}

func zFromSE(diff float64, se float64) float64 {
	if se == 0 {
		if diff > 0 {
			return math.MaxFloat64
		}
		if diff < 0 {
			return -math.MaxFloat64
		}
		return 0
	}
	return diff / se
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func stddev(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	meanValue := mean(values)
	var sumSquares float64
	for _, value := range values {
		delta := value - meanValue
		sumSquares += delta * delta
	}
	return math.Sqrt(sumSquares / float64(len(values)-1))
}

func minValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, value := range values[1:] {
		if value < min {
			min = value
		}
	}
	return min
}

func maxValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	return max
}

func outputHash(output string) string {
	if strings.TrimSpace(output) == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(output))
	return hex.EncodeToString(sum[:])
}

func varianceKeyLess(a VarianceGroupKey, b VarianceGroupKey) bool {
	if a.Layer != b.Layer {
		return a.Layer < b.Layer
	}
	if a.CaseID != b.CaseID {
		return a.CaseID < b.CaseID
	}
	if a.Model != b.Model {
		return a.Model < b.Model
	}
	return a.Scorer < b.Scorer
}
