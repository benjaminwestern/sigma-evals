// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"math"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestVarianceReportFromRunResultSummarizesRepeats(t *testing.T) {
	t.Parallel()

	run := sigmaevals.RunResult{
		SuiteName: "baseline",
		Results: []sigmaevals.CaseResult{
			caseResult("case-1", "model-a", 1, "one", sigmaevals.Score{Name: "exact", Score: 1, Passed: true}),
			caseResult("case-1", "model-a", 2, "two", sigmaevals.Score{Name: "exact", Score: 0, Passed: false}),
			caseResult("case-1", "model-a", 3, "one", sigmaevals.Score{Name: "exact", Score: 1, Passed: true}),
		},
	}
	report := sigmaevals.VarianceReportFromRunResult(run)
	if report.Total.Count != 3 || report.Total.Passed != 2 || report.Total.Failed != 1 {
		t.Fatalf("total = %+v, want 2/3 passing", report.Total)
	}
	if len(report.Groups) != 1 {
		t.Fatalf("groups = %d, want 1", len(report.Groups))
	}
	group := report.Groups[0]
	if group.Key.Layer != sigmaevals.VarianceLayerSuite || group.Key.CaseID != "case-1" || group.Key.Scorer != "exact" {
		t.Fatalf("group key = %+v, want suite/case-1/exact", group.Key)
	}
	if math.Abs(group.PassRate-(2.0/3.0)) > 0.0001 {
		t.Fatalf("pass rate = %.4f, want 0.6667", group.PassRate)
	}
	if math.Abs(group.MeanScore-(2.0/3.0)) > 0.0001 {
		t.Fatalf("mean score = %.4f, want 0.6667", group.MeanScore)
	}
	if group.UniqueOutputs != 2 {
		t.Fatalf("unique outputs = %d, want 2", group.UniqueOutputs)
	}
}

func TestCompareVarianceReportsDetectsRegressionBeyondVariance(t *testing.T) {
	t.Parallel()

	baseline := sigmaevals.BuildVarianceReport("today", []sigmaevals.VarianceSample{
		varianceSample("case-1", "model-a", 1, true),
		varianceSample("case-1", "model-a", 1, true),
		varianceSample("case-1", "model-a", 1, true),
		varianceSample("case-1", "model-a", 1, true),
	})
	current := sigmaevals.BuildVarianceReport("tomorrow", []sigmaevals.VarianceSample{
		varianceSample("case-1", "model-a", 0, false),
		varianceSample("case-1", "model-a", 0, false),
		varianceSample("case-1", "model-a", 0, false),
		varianceSample("case-1", "model-a", 0, false),
	})
	comparison := sigmaevals.CompareVarianceReports(baseline, current, sigmaevals.VarianceCompareOptions{})
	if comparison.Summary.Regressions != 1 || len(comparison.Deltas) != 1 {
		t.Fatalf("comparison = %+v, want one regression", comparison)
	}
	if comparison.Deltas[0].Direction != "regression" || !comparison.Deltas[0].Significant {
		t.Fatalf("delta = %+v, want significant regression", comparison.Deltas[0])
	}
}

func TestCompareVarianceReportsTreatsNoisySmallDeltaAsStable(t *testing.T) {
	t.Parallel()

	baseline := sigmaevals.BuildVarianceReport("today", []sigmaevals.VarianceSample{
		varianceSample("case-1", "model-a", 0.8, true),
		varianceSample("case-1", "model-a", 0.7, true),
		varianceSample("case-1", "model-a", 0.9, true),
		varianceSample("case-1", "model-a", 0.6, true),
	})
	current := sigmaevals.BuildVarianceReport("tomorrow", []sigmaevals.VarianceSample{
		varianceSample("case-1", "model-a", 0.78, true),
		varianceSample("case-1", "model-a", 0.72, true),
		varianceSample("case-1", "model-a", 0.86, true),
		varianceSample("case-1", "model-a", 0.64, true),
	})
	comparison := sigmaevals.CompareVarianceReports(baseline, current, sigmaevals.VarianceCompareOptions{})
	if comparison.Summary.Stable != 1 {
		t.Fatalf("comparison = %+v, want stable", comparison)
	}
	if comparison.Deltas[0].Direction != "stable" || comparison.Deltas[0].Significant {
		t.Fatalf("delta = %+v, want non-significant stable", comparison.Deltas[0])
	}
}

func TestVarianceSamplesCoverBatchJudgeAndJudgeAlignment(t *testing.T) {
	t.Parallel()

	batch := sigmaevals.BatchJudgeResult{Name: "batch", Results: []sigmaevals.JudgeCaseResult{{
		CaseID:       "case-1",
		Target:       "model-a",
		Mode:         sigmaevals.ModeGEval,
		Score:        4.5,
		Passed:       true,
		TargetOutput: "answer",
	}}}
	batchSamples := sigmaevals.VarianceSamplesFromBatchJudgeResult(batch)
	if len(batchSamples) != 1 || batchSamples[0].Layer != sigmaevals.VarianceLayerBatchJudge || batchSamples[0].Score != 4.5 {
		t.Fatalf("batch samples = %+v, want batch judge score sample", batchSamples)
	}

	alignment := sigmaevals.JudgeAlignmentRunResult{Name: "alignment", Results: []sigmaevals.JudgeAlignmentCaseResult{{
		CaseID:         "case-1",
		Model:          "judge-a",
		Provider:       sigma.ProviderOpenAI,
		ExpectedScore:  1,
		ActualScore:    0.8,
		ScoreError:     -0.2,
		ExpectedPassed: true,
		ActualPassed:   true,
		PassedMatch:    true,
	}}}
	alignmentSamples := sigmaevals.VarianceSamplesFromJudgeAlignmentResult(alignment)
	if len(alignmentSamples) != 1 || alignmentSamples[0].Layer != sigmaevals.VarianceLayerJudgeAlignment || alignmentSamples[0].ScoreError != -0.2 {
		t.Fatalf("alignment samples = %+v, want judge alignment score-error sample", alignmentSamples)
	}
}

func caseResult(caseID string, model string, repeat int, output string, score sigmaevals.Score) sigmaevals.CaseResult {
	return sigmaevals.CaseResult{
		CaseID: caseID,
		Model:  model,
		Repeat: repeat,
		Output: output,
		Scores: []sigmaevals.Score{score},
	}
}

func varianceSample(caseID string, model string, score float64, passed bool) sigmaevals.VarianceSample {
	return sigmaevals.VarianceSample{
		Layer:  sigmaevals.VarianceLayerSuite,
		CaseID: caseID,
		Model:  model,
		Scorer: "exact",
		Score:  score,
		Passed: passed,
	}
}
