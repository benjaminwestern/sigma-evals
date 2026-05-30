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
	"sync"
	"time"

	"github.com/wintermi/sigma"
)

// JudgeAlignmentCase is one human-labelled example for evaluating a judge.
type JudgeAlignmentCase struct {
	ID             string         `json:"id"`
	Input          string         `json:"input,omitempty"`
	TargetOutput   string         `json:"targetOutput"`
	GroundTruth    string         `json:"groundTruth,omitempty"`
	Rubric         string         `json:"rubric,omitempty"`
	ExpectedScore  float64        `json:"expectedScore"`
	ExpectedPassed bool           `json:"expectedPassed"`
	Tags           []string       `json:"tags,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

// JudgeAlignmentSpec configures a judge-quality eval against labelled cases.
type JudgeAlignmentSpec struct {
	Name          string               `json:"name"`
	Version       string               `json:"version,omitempty"`
	Cases         []JudgeAlignmentCase `json:"cases"`
	JudgeTargets  []Target             `json:"judgeTargets,omitempty"`
	JudgeModels   []sigma.Model        `json:"judgeModels"`
	Mode          Mode                 `json:"mode,omitempty"`
	PassThreshold float64              `json:"passThreshold,omitempty"`
	Options       []sigma.Option       `json:"-"`
	Tolerance     float64              `json:"tolerance,omitempty"`
	Concurrency   int                  `json:"concurrency,omitempty"`
}

// JudgeAlignmentRunResult records one judge alignment run.
type JudgeAlignmentRunResult struct {
	Name      string                       `json:"name"`
	Version   string                       `json:"version,omitempty"`
	StartedAt time.Time                    `json:"startedAt"`
	EndedAt   time.Time                    `json:"endedAt"`
	Results   []JudgeAlignmentCaseResult   `json:"results"`
	Summary   JudgeAlignmentSummary        `json:"summary"`
	ByModel   map[string]JudgeModelSummary `json:"byModel,omitempty"`
}

// JudgeAlignmentCaseResult records one labelled case judged by one judge model.
type JudgeAlignmentCaseResult struct {
	CaseID          string           `json:"caseId"`
	Model           string           `json:"model"`
	Provider        sigma.ProviderID `json:"provider,omitempty"`
	ExpectedScore   float64          `json:"expectedScore"`
	ActualScore     float64          `json:"actualScore,omitempty"`
	ScoreError      float64          `json:"scoreError,omitempty"`
	ExpectedPassed  bool             `json:"expectedPassed"`
	ActualPassed    bool             `json:"actualPassed,omitempty"`
	PassedMatch     bool             `json:"passedMatch"`
	WithinTolerance bool             `json:"withinTolerance"`
	Result          *JudgeResult     `json:"result,omitempty"`
	Error           string           `json:"error,omitempty"`
	DurationMS      int64            `json:"durationMs"`
}

// JudgeAlignmentSummary aggregates judge alignment results.
type JudgeAlignmentSummary struct {
	Total                int                   `json:"total"`
	Errors               int                   `json:"errors"`
	ScoreWithinTolerance int                   `json:"scoreWithinTolerance"`
	ToleranceAccuracy    float64               `json:"toleranceAccuracy,omitempty"`
	Regression           RegressionMetrics     `json:"regression"`
	Classification       ClassificationMetrics `json:"classification"`
	Calibration          CalibrationMetrics    `json:"calibration"`
}

// JudgeModelSummary aggregates judge alignment results for one judge model.
type JudgeModelSummary struct {
	Total                int                   `json:"total"`
	Errors               int                   `json:"errors"`
	ScoreWithinTolerance int                   `json:"scoreWithinTolerance"`
	ToleranceAccuracy    float64               `json:"toleranceAccuracy,omitempty"`
	Regression           RegressionMetrics     `json:"regression"`
	Classification       ClassificationMetrics `json:"classification"`
	Calibration          CalibrationMetrics    `json:"calibration"`
	DurationMS           int64                 `json:"durationMs"`
}

// EvaluateJudges runs judge models against human-labelled judge cases.
func (e *Evaluator) EvaluateJudges(ctx context.Context, spec JudgeAlignmentSpec) (JudgeAlignmentRunResult, error) {
	if strings.TrimSpace(spec.Name) == "" {
		return JudgeAlignmentRunResult{}, fmt.Errorf("judge alignment name is required")
	}
	if len(spec.Cases) == 0 {
		return JudgeAlignmentRunResult{}, fmt.Errorf("judge alignment spec must contain at least one case")
	}
	judgeTargets := judgeAlignmentTargets(spec)
	if len(judgeTargets) == 0 {
		return JudgeAlignmentRunResult{}, fmt.Errorf("at least one judge target is required")
	}
	concurrency := spec.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}
	tolerance := spec.Tolerance
	if tolerance <= 0 {
		tolerance = 0.5
	}

	startedAt := time.Now().UTC()
	jobs := make(chan judgeAlignmentJob)
	results := make([]JudgeAlignmentCaseResult, 0, len(spec.Cases)*len(judgeTargets))
	var mu sync.Mutex

	worker := func() {
		for job := range jobs {
			result := e.evaluateJudgeCase(ctx, spec, job, tolerance)
			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}
	}

	workerCount := concurrency
	totalJobs := len(spec.Cases) * len(judgeTargets)
	if workerCount > totalJobs {
		workerCount = totalJobs
	}
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	for _, c := range spec.Cases {
		if strings.TrimSpace(c.ID) == "" {
			close(jobs)
			wg.Wait()
			return JudgeAlignmentRunResult{}, fmt.Errorf("judge alignment case id is required")
		}
		for _, target := range judgeTargets {
			select {
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return JudgeAlignmentRunResult{}, ctx.Err()
			case jobs <- judgeAlignmentJob{Case: c, Target: target, Model: target.modelForScoring()}:
			}
		}
	}
	close(jobs)
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].CaseID != results[j].CaseID {
			return results[i].CaseID < results[j].CaseID
		}
		return results[i].Model < results[j].Model
	})
	summary, byModel := summarizeJudgeAlignment(results)
	return JudgeAlignmentRunResult{
		Name:      spec.Name,
		Version:   spec.Version,
		StartedAt: startedAt,
		EndedAt:   time.Now().UTC(),
		Results:   results,
		Summary:   summary,
		ByModel:   byModel,
	}, nil
}

func judgeAlignmentTargets(spec JudgeAlignmentSpec) []Target {
	if len(spec.JudgeTargets) > 0 {
		out := make([]Target, 0, len(spec.JudgeTargets))
		for _, target := range spec.JudgeTargets {
			out = append(out, targetWithModelFallback(target, sigma.Model{}))
		}
		return out
	}
	out := make([]Target, 0, len(spec.JudgeModels))
	for _, model := range spec.JudgeModels {
		out = append(out, TargetFromModel(model))
	}
	return out
}

type judgeAlignmentJob struct {
	Case   JudgeAlignmentCase
	Target Target
	Model  sigma.Model
}

func (e *Evaluator) evaluateJudgeCase(ctx context.Context, spec JudgeAlignmentSpec, job judgeAlignmentJob, tolerance float64) JudgeAlignmentCaseResult {
	startedAt := time.Now()
	result := JudgeAlignmentCaseResult{
		CaseID:         job.Case.ID,
		Model:          modelLabel(job.Model),
		Provider:       job.Model.Provider,
		ExpectedScore:  job.Case.ExpectedScore,
		ExpectedPassed: job.Case.ExpectedPassed,
	}

	judgeResult, err := e.Judge(ctx, JudgeInput{
		Input:         job.Case.Input,
		TargetOutput:  job.Case.TargetOutput,
		GroundTruth:   job.Case.GroundTruth,
		Rubric:        job.Case.Rubric,
		Judge:         job.Target,
		JudgeModel:    job.Model,
		Mode:          spec.Mode,
		PassThreshold: spec.PassThreshold,
		JudgeOptions:  spec.Options,
	})
	result.DurationMS = time.Since(startedAt).Milliseconds()
	if err != nil {
		result.Error = err.Error()
		return result
	}

	result.Result = &judgeResult
	result.ActualScore = judgeResult.Score
	result.ScoreError = judgeResult.Score - job.Case.ExpectedScore
	result.ActualPassed = judgeResult.Passed
	result.PassedMatch = judgeResult.Passed == job.Case.ExpectedPassed
	result.WithinTolerance = math.Abs(result.ScoreError) <= tolerance
	return result
}

func summarizeJudgeAlignment(results []JudgeAlignmentCaseResult) (JudgeAlignmentSummary, map[string]JudgeModelSummary) {
	var summary JudgeAlignmentSummary
	byModel := make(map[string]JudgeModelSummary)

	var expectedScores []float64
	var actualScores []float64
	var expectedPassed []bool
	var actualPassed []bool
	modelExpectedScores := map[string][]float64{}
	modelActualScores := map[string][]float64{}
	modelExpectedPassed := map[string][]bool{}
	modelActualPassed := map[string][]bool{}

	for _, result := range results {
		summary.Total++
		modelSummary := byModel[result.Model]
		modelSummary.Total++
		modelSummary.DurationMS += result.DurationMS
		if result.Error != "" {
			summary.Errors++
			modelSummary.Errors++
			byModel[result.Model] = modelSummary
			continue
		}

		expectedScores = append(expectedScores, result.ExpectedScore)
		actualScores = append(actualScores, result.ActualScore)
		expectedPassed = append(expectedPassed, result.ExpectedPassed)
		actualPassed = append(actualPassed, result.ActualPassed)
		modelExpectedScores[result.Model] = append(modelExpectedScores[result.Model], result.ExpectedScore)
		modelActualScores[result.Model] = append(modelActualScores[result.Model], result.ActualScore)
		modelExpectedPassed[result.Model] = append(modelExpectedPassed[result.Model], result.ExpectedPassed)
		modelActualPassed[result.Model] = append(modelActualPassed[result.Model], result.ActualPassed)

		if result.WithinTolerance {
			summary.ScoreWithinTolerance++
			modelSummary.ScoreWithinTolerance++
		}
		byModel[result.Model] = modelSummary
	}

	valid := len(actualScores)
	if valid > 0 {
		summary.ToleranceAccuracy = float64(summary.ScoreWithinTolerance) / float64(valid)
		summary.Regression = ComputeRegressionMetrics(expectedScores, actualScores)
		summary.Classification = ComputeClassificationMetrics(expectedPassed, actualPassed)
		summary.Calibration = ComputeCalibrationMetrics(expectedPassed, actualScores, minFloat64(expectedScores), maxFloat64(expectedScores))
	}
	for model, modelSummary := range byModel {
		modelValid := len(modelActualScores[model])
		if modelValid > 0 {
			modelSummary.ToleranceAccuracy = float64(modelSummary.ScoreWithinTolerance) / float64(modelValid)
			modelSummary.Regression = ComputeRegressionMetrics(modelExpectedScores[model], modelActualScores[model])
			modelSummary.Classification = ComputeClassificationMetrics(modelExpectedPassed[model], modelActualPassed[model])
			modelSummary.Calibration = ComputeCalibrationMetrics(modelExpectedPassed[model], modelActualScores[model], minFloat64(modelExpectedScores[model]), maxFloat64(modelExpectedScores[model]))
		}
		byModel[model] = modelSummary
	}
	return summary, byModel
}

func minFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	minValue := values[0]
	for _, value := range values[1:] {
		if value < minValue {
			minValue = value
		}
	}
	return minValue
}

func maxFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}
