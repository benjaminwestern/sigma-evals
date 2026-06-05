// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestRegisterCommonProviders(t *testing.T) {
	t.Parallel()

	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		t.Fatal(err)
	}
	for _, provider := range []sigma.ProviderID{sigma.ProviderAnthropic, sigma.ProviderGoogle, sigma.ProviderGoogleVertex, sigma.ProviderMistral, sigma.ProviderAmazonBedrock, sigma.ProviderXAI} {
		if _, ok := registry.TextProvider(provider); !ok {
			t.Fatalf("text provider %q was not registered", provider)
		}
	}
	if _, ok := registry.ImageProvider(sigma.ProviderOpenRouter); !ok {
		t.Fatal("openrouter image provider was not registered")
	}
}

func TestParseCacheRetention(t *testing.T) {
	t.Parallel()

	retention, err := parseCacheRetention("persistent")
	if err != nil {
		t.Fatal(err)
	}
	if retention != sigma.CacheRetentionPersistent {
		t.Fatalf("retention = %q, want persistent", retention)
	}
	if _, err := parseCacheRetention("forever"); err == nil {
		t.Fatal("parseCacheRetention accepted unknown value")
	}
}

func TestRequestOptionsAcceptSessionAndCacheRetention(t *testing.T) {
	t.Parallel()

	options, err := requestOptions(" suite-run ", "long")
	if err != nil {
		t.Fatal(err)
	}
	var applied sigma.Options
	for _, option := range options {
		option(&applied)
	}
	if applied.SessionID != "suite-run" {
		t.Fatalf("session id = %q, want suite-run", applied.SessionID)
	}
	if applied.CacheRetention != sigma.CacheRetentionLong {
		t.Fatalf("cache retention = %q, want long", applied.CacheRetention)
	}
}

func TestListRubricsCommandWritesBuiltIns(t *testing.T) {
	t.Parallel()

	out := filepath.Join(t.TempDir(), "rubrics.json")
	if err := run([]string{"list-rubrics", "--out", out}); err != nil {
		t.Fatal(err)
	}
	var rubrics []sigmaevals.StandardRubric
	readJSONFileForTest(t, out, &rubrics)
	if len(rubrics) == 0 || rubrics[0].ID == "" {
		t.Fatalf("rubrics = %+v, want built-ins", rubrics)
	}
	byID := map[string]sigmaevals.StandardRubric{}
	for _, rubric := range rubrics {
		byID[rubric.ID] = rubric
	}
	accuracy := byID["accuracy"]
	if accuracy.Prompt == "" || !containsString(accuracy.Aliases, "rubric-accuracy") {
		t.Fatalf("accuracy rubric = %+v, want prompt and compatibility alias", accuracy)
	}
}

func TestVarianceReportCommandReadsSuiteResult(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	input := filepath.Join(dir, "run.json")
	out := filepath.Join(dir, "variance.json")
	runResult := sigmaevals.RunResult{SuiteName: "suite", Results: []sigmaevals.CaseResult{{
		CaseID: "case-1",
		Model:  "model-a",
		Repeat: 1,
		Scores: []sigmaevals.Score{{Name: "exact", Score: 1, Passed: true}},
	}}}
	writeJSONFileForTest(t, input, runResult)
	if err := run([]string{"variance-report", "--layer", "suite", "--input", input, "--out", out}); err != nil {
		t.Fatal(err)
	}
	var report sigmaevals.VarianceReport
	readJSONFileForTest(t, out, &report)
	if report.Total.Count != 1 || len(report.Groups) != 1 {
		t.Fatalf("report = %+v, want one sample group", report)
	}
}

func TestVarianceReportCommandReadsSamplesJSONL(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	input := filepath.Join(dir, "samples.jsonl")
	out := filepath.Join(dir, "variance.json")
	file, err := os.Create(input)
	if err != nil {
		t.Fatal(err)
	}
	err = sigmaevals.WriteVarianceSamplesJSONL(file, []sigmaevals.VarianceSample{
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 1, Passed: true},
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 0, Passed: false},
	})
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"variance-report", "--layer", "samples", "--input", input, "--out", out}); err != nil {
		t.Fatal(err)
	}
	var report sigmaevals.VarianceReport
	readJSONFileForTest(t, out, &report)
	if report.Total.Count != 2 || report.Total.Passed != 1 || report.Total.Failed != 1 {
		t.Fatalf("report = %+v, want mixed samples summarized", report)
	}
}

func TestCompareVarianceCommandWritesComparison(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")
	currentPath := filepath.Join(dir, "current.json")
	out := filepath.Join(dir, "comparison.json")
	samples := []sigmaevals.VarianceSample{{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 1, Passed: true}}
	writeJSONFileForTest(t, baselinePath, sigmaevals.BuildVarianceReport("baseline", samples))
	writeJSONFileForTest(t, currentPath, sigmaevals.BuildVarianceReport("current", samples))
	if err := run([]string{"compare-variance", "--baseline", baselinePath, "--current", currentPath, "--out", out}); err != nil {
		t.Fatal(err)
	}
	var comparison sigmaevals.VarianceComparison
	readJSONFileForTest(t, out, &comparison)
	if comparison.Summary.Stable != 1 {
		t.Fatalf("comparison = %+v, want stable", comparison)
	}
}

func TestCompareVarianceCommandFailsOnRegression(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")
	currentPath := filepath.Join(dir, "current.json")
	out := filepath.Join(dir, "comparison.json")
	baseline := sigmaevals.BuildVarianceReport("baseline", []sigmaevals.VarianceSample{
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 1, Passed: true},
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 1, Passed: true},
	})
	current := sigmaevals.BuildVarianceReport("current", []sigmaevals.VarianceSample{
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 0, Passed: false},
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 0, Passed: false},
	})
	writeJSONFileForTest(t, baselinePath, baseline)
	writeJSONFileForTest(t, currentPath, current)
	if err := run([]string{"compare-variance", "--baseline", baselinePath, "--current", currentPath, "--out", out}); err == nil {
		t.Fatal("compare-variance succeeded, want nonzero regression error")
	}
	var comparison sigmaevals.VarianceComparison
	readJSONFileForTest(t, out, &comparison)
	if comparison.Summary.Regressions != 1 {
		t.Fatalf("comparison = %+v, want one regression", comparison)
	}
}

func writeJSONFileForTest(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func readJSONFileForTest(t *testing.T, path string, value any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, value); err != nil {
		t.Fatal(err)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
