// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
)

const (
	outputJSON  = "json"
	outputJSONL = "jsonl"
	outputMD    = "md"
)

func writeRenderedOutput(w io.Writer, format string, value any) error {
	data, err := renderOutput(format, value)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

func renderOutput(format string, value any) ([]byte, error) {
	switch normalizeOutputFormat(format) {
	case outputJSON:
		data, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			return nil, err
		}
		return append(data, '\n'), nil
	case outputJSONL:
		return renderJSONL(value)
	case outputMD:
		return []byte(renderMarkdown(value)), nil
	default:
		return nil, fmt.Errorf("unknown output format %q", format)
	}
}

func normalizeOutputFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "json":
		return outputJSON
	case "jsonl", "ndjson":
		return outputJSONL
	case "md", "markdown":
		return outputMD
	default:
		return strings.ToLower(strings.TrimSpace(format))
	}
}

func renderJSONL(value any) ([]byte, error) {
	items := jsonlItems(value)
	var b bytes.Buffer
	encoder := json.NewEncoder(&b)
	for _, item := range items {
		if err := encoder.Encode(item); err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}

func jsonlItems(value any) []any {
	switch v := value.(type) {
	case smokeSummary:
		items := make([]any, 0, len(v.Suites))
		for _, item := range v.Suites {
			items = append(items, item)
		}
		return items
	case sigmaevals.RunResult:
		items := make([]any, 0, len(v.Results))
		for _, item := range v.Results {
			items = append(items, item)
		}
		return items
	case sigmaevals.JudgeResult:
		return []any{v}
	case sigmaevals.BatchJudgeResult:
		items := make([]any, 0, len(v.Results))
		for _, item := range v.Results {
			items = append(items, item)
		}
		return items
	case sigmaevals.JudgeAlignmentRunResult:
		items := make([]any, 0, len(v.Results))
		for _, item := range v.Results {
			items = append(items, item)
		}
		return items
	case []sigmaevals.StandardRubric:
		items := make([]any, 0, len(v))
		for _, item := range v {
			items = append(items, item)
		}
		return items
	case sigmaevals.VarianceReport:
		items := make([]any, 0, len(v.Groups))
		for _, item := range v.Groups {
			items = append(items, item)
		}
		return items
	case sigmaevals.VarianceComparison:
		items := make([]any, 0, len(v.Deltas))
		for _, item := range v.Deltas {
			items = append(items, item)
		}
		return items
	default:
		return []any{value}
	}
}

func renderMarkdown(value any) string {
	switch v := value.(type) {
	case smokeSummary:
		return markdownSmokeSummary(v)
	case sigmaevals.RunResult:
		return markdownRunResult(v)
	case sigmaevals.JudgeResult:
		return markdownJudgeResult(v)
	case sigmaevals.BatchJudgeResult:
		return markdownBatchJudgeResult(v)
	case sigmaevals.JudgeAlignmentRunResult:
		return markdownJudgeAlignmentRunResult(v)
	case []sigmaevals.StandardRubric:
		return markdownRubrics(v)
	case sigmaevals.VarianceReport:
		return markdownVarianceReport(v)
	case sigmaevals.VarianceComparison:
		return markdownVarianceComparison(v)
	default:
		data, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			return fmt.Sprintf("```text\n%v\n```\n", value)
		}
		return "```json\n" + string(data) + "\n```\n"
	}
}

func markdownSmokeSummary(summary smokeSummary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Smoke examples\n\n")
	fmt.Fprintf(&b, "Passed **%d/%d** suites/cases, failed **%d**, errors **%d**.\n\n", summary.Passed, summary.Total, summary.Failed, summary.Errors)
	b.WriteString("| Suite | Passed | Failed | Errors | Total |\n")
	b.WriteString("| --- | ---: | ---: | ---: | ---: |\n")
	for _, suite := range summary.Suites {
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d |\n", mdCell(suite.SuiteName), suite.Summary.Passed, suite.Summary.Failed, suite.Summary.Errors, suite.Summary.Total)
	}
	return b.String()
}

func markdownRunResult(result sigmaevals.RunResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Suite run: %s\n\n", mdText(result.SuiteName))
	fmt.Fprintf(&b, "Passed **%d/%d**, failed **%d**, errors **%d**, score count **%d**.\n\n", result.Summary.Passed, result.Summary.Total, result.Summary.Failed, result.Summary.Errors, result.Summary.ScoreCount)
	if len(result.ByModel) > 0 {
		b.WriteString("## By model\n\n")
		b.WriteString("| Model | Passed | Failed | Errors | Mean score | Total |\n")
		b.WriteString("| --- | ---: | ---: | ---: | ---: | ---: |\n")
		for _, model := range sortedModelSummaryKeys(result.ByModel) {
			summary := result.ByModel[model]
			fmt.Fprintf(&b, "| %s | %d | %d | %d | %.3f | %d |\n", mdCell(model), summary.Passed, summary.Failed, summary.Errors, summary.MeanScore, summary.Total)
		}
		b.WriteString("\n")
	}
	b.WriteString("## Failing cases\n\n")
	b.WriteString("| Case | Model | Repeat | Scores | Error | Output |\n")
	b.WriteString("| --- | --- | ---: | --- | --- | --- |\n")
	written := 0
	for _, item := range result.Results {
		if item.Error == "" && caseResultPassed(item) {
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %d | %s | %s | %s |\n", mdCell(item.CaseID), mdCell(item.Model), item.Repeat, mdCell(scoreList(item.Scores)), mdCell(item.Error), mdCell(snippet(item.Output, 96)))
		written++
	}
	if written == 0 {
		b.WriteString("| _none_ |  |  |  |  |  |\n")
	}
	return b.String()
}

func markdownJudgeResult(result sigmaevals.JudgeResult) string {
	var b strings.Builder
	b.WriteString("# Judge output\n\n")
	fmt.Fprintf(&b, "| Mode | Score | Passed | Pass threshold |\n")
	b.WriteString("| --- | ---: | --- | ---: |\n")
	fmt.Fprintf(&b, "| %s | %.3f | %t | %.3f |\n\n", mdCell(string(result.Mode)), result.Score, result.Passed, result.PassThreshold)
	if strings.TrimSpace(result.Rationale) != "" {
		b.WriteString("## Rationale\n\n")
		b.WriteString(mdText(result.Rationale))
		b.WriteString("\n")
	}
	return b.String()
}

func markdownBatchJudgeResult(result sigmaevals.BatchJudgeResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Batch judge: %s\n\n", mdText(result.Name))
	fmt.Fprintf(&b, "Passed **%d/%d**, failed **%d**, errors **%d**, mean score **%.3f**.\n\n", result.Summary.Passed, result.Summary.Total, result.Summary.Failed, result.Summary.Errors, result.Summary.MeanScore)
	b.WriteString("| Case | Repeat | Score | Passed | Error | Output | Rationale |\n")
	b.WriteString("| --- | ---: | ---: | --- | --- | --- | --- |\n")
	for _, item := range result.Results {
		if item.Passed && item.Error == "" {
			continue
		}
		fmt.Fprintf(&b, "| %s | %d | %.3f | %t | %s | %s | %s |\n", mdCell(item.CaseID), item.Repeat, item.Score, item.Passed, mdCell(item.Error), mdCell(snippet(item.TargetOutput, 80)), mdCell(snippet(item.Rationale, 120)))
	}
	return b.String()
}

func markdownJudgeAlignmentRunResult(result sigmaevals.JudgeAlignmentRunResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Judge alignment: %s\n\n", mdText(result.Name))
	fmt.Fprintf(&b, "Total **%d**, errors **%d**, tolerance accuracy **%.3f**, classification accuracy **%.3f**.\n\n", result.Summary.Total, result.Summary.Errors, result.Summary.ToleranceAccuracy, result.Summary.Classification.Accuracy)
	b.WriteString("| Case | Model | Expected | Actual | Passed match | Within tolerance | Error |\n")
	b.WriteString("| --- | --- | ---: | ---: | --- | --- | --- |\n")
	for _, item := range result.Results {
		if item.Error == "" && item.PassedMatch && item.WithinTolerance {
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %.3f | %.3f | %t | %t | %s |\n", mdCell(item.CaseID), mdCell(item.Model), item.ExpectedScore, item.ActualScore, item.PassedMatch, item.WithinTolerance, mdCell(item.Error))
	}
	return b.String()
}

func markdownRubrics(rubrics []sigmaevals.StandardRubric) string {
	var b strings.Builder
	b.WriteString("# Built-in rubrics\n\n")
	b.WriteString("| ID | Name | Aliases | Description |\n")
	b.WriteString("| --- | --- | --- | --- |\n")
	for _, rubric := range rubrics {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", mdCell(rubric.ID), mdCell(rubric.Name), mdCell(strings.Join(rubric.Aliases, ", ")), mdCell(rubric.Description))
	}
	return b.String()
}

func markdownVarianceReport(report sigmaevals.VarianceReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Variance report: %s\n\n", mdText(report.Name))
	fmt.Fprintf(&b, "Groups **%d**, pass rate **%.3f**, mean score **%.3f**, stddev **%.3f**, stderr **%.3f**, unique outputs **%d**, errors **%d**.\n\n", len(report.Groups), report.Total.PassRate, report.Total.MeanScore, report.Total.StdDevScore, report.Total.StdErrScore, report.Total.UniqueOutputs, report.Total.Errors)
	b.WriteString("| Case | Model | Scorer | N | Pass rate | Mean | Stddev | Stderr | Unique outputs | Errors | Note |\n")
	b.WriteString("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
	for _, group := range report.Groups {
		note := ""
		if group.Errors > 0 {
			note = "errors"
		} else if group.Passed+group.Failed > 0 && group.PassRate == 0 {
			note = "stable failure"
		} else if group.Passed+group.Failed > 0 && group.PassRate == 1 && group.StdDevScore == 0 {
			note = "stable pass"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %d | %d | %s |\n", mdCell(group.Key.CaseID), mdCell(valueOrDash(group.Key.Model)), mdCell(valueOrDash(group.Key.Scorer)), group.Count, group.PassRate, group.MeanScore, group.StdDevScore, group.StdErrScore, group.UniqueOutputs, group.Errors, mdCell(note))
	}
	return b.String()
}

func markdownVarianceComparison(comparison sigmaevals.VarianceComparison) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Variance comparison: %s vs %s\n\n", mdText(comparison.BaselineName), mdText(comparison.CurrentName))
	fmt.Fprintf(&b, "Regressions **%d**, improvements **%d**, stable **%d**, new **%d**, missing **%d**.\n\n", comparison.Summary.Regressions, comparison.Summary.Improvements, comparison.Summary.Stable, comparison.Summary.New, comparison.Summary.Missing)
	b.WriteString("| Direction | Case | Model | Scorer | Pass delta | Score delta | Score z | Pass z | Significant |\n")
	b.WriteString("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n")
	for _, delta := range comparison.Deltas {
		if delta.Direction == "stable" {
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %t |\n", mdCell(delta.Direction), mdCell(delta.Key.CaseID), mdCell(valueOrDash(delta.Key.Model)), mdCell(valueOrDash(delta.Key.Scorer)), delta.DeltaPassRate, delta.DeltaMeanScore, delta.ScoreZ, delta.PassRateZ, delta.Significant)
	}
	return b.String()
}

func mdCell(value string) string {
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "|", "\\|")
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	return value
}

func mdText(value string) string {
	return strings.TrimSpace(value)
}

func snippet(value string, limit int) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) <= limit {
		return value
	}
	return value[:limit-1] + "…"
}

func scoreList(scores []sigmaevals.Score) string {
	if len(scores) == 0 {
		return ""
	}
	parts := make([]string, 0, len(scores))
	for _, score := range scores {
		parts = append(parts, fmt.Sprintf("%s=%.3f passed=%t", score.Name, score.Score, score.Passed))
	}
	return strings.Join(parts, "; ")
}

func caseResultPassed(result sigmaevals.CaseResult) bool {
	if result.Error != "" || len(result.Scores) == 0 {
		return false
	}
	for _, score := range result.Scores {
		if !score.Passed {
			return false
		}
	}
	return true
}

func sortedModelSummaryKeys(values map[string]sigmaevals.ModelSummary) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
