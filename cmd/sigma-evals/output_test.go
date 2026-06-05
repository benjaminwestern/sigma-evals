// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
)

func TestRenderMarkdownVarianceReportHighlightsStableFailures(t *testing.T) {
	t.Parallel()

	report := sigmaevals.BuildVarianceReport("repeat-check", []sigmaevals.VarianceSample{
		{Layer: sigmaevals.VarianceLayerBatchJudge, CaseID: "reverse-sigma", Model: "deepseek", Scorer: "evaluate", Score: 0, Passed: false},
		{Layer: sigmaevals.VarianceLayerBatchJudge, CaseID: "reverse-sigma", Model: "deepseek", Scorer: "evaluate", Score: 0, Passed: false},
	})
	data, err := renderOutput(outputMD, report)
	if err != nil {
		t.Fatal(err)
	}
	markdown := string(data)
	for _, want := range []string{"# Variance report: repeat-check", "reverse-sigma", "stable failure", "| Case | Model | Scorer |"} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("markdown missing %q:\n%s", want, markdown)
		}
	}
}

func TestRenderJSONLBatchJudgeOutputsCaseRows(t *testing.T) {
	t.Parallel()

	result := sigmaevals.BatchJudgeResult{Name: "batch", Results: []sigmaevals.JudgeCaseResult{
		{CaseID: "case-1", Repeat: 1, Score: 1, Passed: true},
		{CaseID: "case-1", Repeat: 2, Score: 0, Passed: false},
	}}
	data, err := renderOutput(outputJSONL, result)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 || !strings.Contains(lines[1], `"repeat":2`) || !strings.Contains(lines[1], `"passed":false`) {
		t.Fatalf("jsonl = %q, want one case result per line", string(data))
	}
}
