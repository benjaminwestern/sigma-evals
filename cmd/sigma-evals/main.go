// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/anthropic"
	"github.com/wintermi/sigma/provider/bedrock"
	"github.com/wintermi/sigma/provider/fireworks"
	"github.com/wintermi/sigma/provider/google"
	"github.com/wintermi/sigma/provider/mistral"
	"github.com/wintermi/sigma/provider/openai"
	"github.com/wintermi/sigma/provider/opencode"
	"github.com/wintermi/sigma/provider/openrouter"
	"github.com/wintermi/sigma/provider/xai"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	switch args[0] {
	case "smoke-examples":
		return runSmokeExamples(args[1:])
	case "run-suite":
		return runSuite(args[1:])
	case "judge-output":
		return runJudgeOutput(args[1:])
	case "judge-batch":
		return runJudgeBatch(args[1:])
	case "judge-alignment":
		return runJudgeAlignment(args[1:])
	case "list-rubrics":
		return runListRubrics(args[1:])
	case "variance-report":
		return runVarianceReport(args[1:])
	case "compare-variance":
		return runCompareVariance(args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `sigma-evals is a small SDK consumer for the sigmaevals interfaces.

Commands:
  smoke-examples    Run all JSON examples with a scripted TargetCompleter, no network.
  run-suite         Run one suite against real Sigma targets.
  judge-output      Score an existing output with an LLM judge target.
  judge-batch       Score many target or saved outputs with an LLM judge.
  judge-alignment   Evaluate judge quality against labelled examples.
  list-rubrics      Print built-in rubric IDs, aliases, and prompts.
  variance-report   Build a variance report from suite, batch, alignment, or samples JSON/JSONL.
  compare-variance  Compare baseline/current variance reports.

Examples:
  sigma-evals smoke-examples --examples examples --out runs/smoke.json
  sigma-evals run-suite --suite examples/generic/answer-aliases.json --target fireworks=accounts/fireworks/routers/kimi-k2p6-turbo --repeat 100 --samples-out runs/today.jsonl
  sigma-evals judge-output --judge openai=gpt-4o --target-output "Bonjour" --ground-truth "Bonjour" --rubric accuracy
  sigma-evals variance-report --layer suite --input runs/today.json --out runs/today-variance.json
  sigma-evals compare-variance --baseline runs/today-variance.json --current runs/tomorrow-variance.json`)
}

func runSmokeExamples(args []string) error {
	flags := flag.NewFlagSet("smoke-examples", flag.ContinueOnError)
	examplesDir := flags.String("examples", "examples", "directory containing example JSON suites")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	if err := flags.Parse(args); err != nil {
		return err
	}
	paths, err := exampleSuitePaths(*examplesDir)
	if err != nil {
		return err
	}
	results := make([]sigmaevals.RunResult, 0, len(paths))
	for _, path := range paths {
		suite, err := sigmaevals.LoadSuiteFile(path)
		if err != nil {
			return fmt.Errorf("load %s: %w", path, err)
		}
		completer, err := scriptedCompleterForSuite(suite)
		if err != nil {
			return fmt.Errorf("script %s: %w", path, err)
		}
		result, err := sigmaevals.NewTargetRunner(completer).Run(context.Background(), sigmaevals.TargetRunSpec{
			Suite:       suite,
			Targets:     []sigmaevals.Target{{Provider: "scripted", ModelID: "example", Label: "scripted/example"}},
			Scorers:     []sigmaevals.Scorer{sigmaevals.AutoScorer{}},
			Concurrency: 1,
		})
		if err != nil {
			return fmt.Errorf("run %s: %w", path, err)
		}
		results = append(results, result)
	}
	summary := smokeSummary{Suites: results, StartedAt: time.Now().UTC()}
	for _, result := range results {
		summary.Total += result.Summary.Total
		summary.Passed += result.Summary.Passed
		summary.Failed += result.Summary.Failed
		summary.Errors += result.Summary.Errors
	}
	summary.EndedAt = time.Now().UTC()
	if err := writeOutput(*outPath, *outFormat, summary); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "smoke examples: %d/%d passed, failed=%d, errors=%d\n", summary.Passed, summary.Total, summary.Failed, summary.Errors)
	if summary.Failed > 0 || summary.Errors > 0 {
		return errors.New("example smoke failed")
	}
	return nil
}

type smokeSummary struct {
	StartedAt time.Time              `json:"startedAt"`
	EndedAt   time.Time              `json:"endedAt"`
	Total     int                    `json:"total"`
	Passed    int                    `json:"passed"`
	Failed    int                    `json:"failed"`
	Errors    int                    `json:"errors"`
	Suites    []sigmaevals.RunResult `json:"suites"`
}

func runSuite(args []string) error {
	flags := flag.NewFlagSet("run-suite", flag.ContinueOnError)
	suitePath := flags.String("suite", "", "JSON suite file")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	samplesOutPath := flags.String("samples-out", "", "optional variance samples JSONL output path")
	repeats := flags.Int("repeat", 1, "number of repeats per target")
	concurrency := flags.Int("concurrency", 1, "number of concurrent target attempts")
	sessionID := flags.String("session-id", "", "optional Sigma session ID for provider affinity and prompt-cache keys")
	cacheRetention := flags.String("cache-retention", "", "optional provider prompt-cache retention: none, short, long, ephemeral, or persistent")
	var targetFlags repeatedString
	flags.Var(&targetFlags, "target", "target as provider=model; repeatable")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*suitePath) == "" {
		return fmt.Errorf("--suite is required")
	}
	if len(targetFlags) == 0 {
		return fmt.Errorf("at least one --target is required")
	}
	suite, err := sigmaevals.LoadSuiteFile(*suitePath)
	if err != nil {
		return err
	}
	targets := make([]sigmaevals.Target, 0, len(targetFlags))
	for _, raw := range targetFlags {
		target, err := sigmaevals.ParseTarget(raw)
		if err != nil {
			return err
		}
		targets = append(targets, target)
	}
	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		return err
	}
	options, err := requestOptions(*sessionID, *cacheRetention)
	if err != nil {
		return err
	}
	client := sigma.NewClient(sigma.WithRegistry(registry))
	result, err := sigmaevals.NewTargetRunner(sigmaevals.SigmaTargetCompleter{Client: client, Registry: registry}).Run(context.Background(), sigmaevals.TargetRunSpec{
		Suite:       suite,
		Targets:     targets,
		Scorers:     []sigmaevals.Scorer{sigmaevals.AutoScorer{}},
		Options:     options,
		Repeats:     *repeats,
		Concurrency: *concurrency,
	})
	if err != nil {
		return err
	}
	if err := writeOutput(*outPath, *outFormat, result); err != nil {
		return err
	}
	if err := writeSamplesJSONL(*samplesOutPath, sigmaevals.VarianceSamplesFromRunResult(result)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %d/%d passed, failed=%d, errors=%d\n", result.SuiteName, result.Summary.Passed, result.Summary.Total, result.Summary.Failed, result.Summary.Errors)
	if result.Summary.Failed > 0 || result.Summary.Errors > 0 {
		return errors.New("suite run failed")
	}
	return nil
}

func runJudgeOutput(args []string) error {
	flags := flag.NewFlagSet("judge-output", flag.ContinueOnError)
	judgeRaw := flags.String("judge", "", "judge target as provider=model")
	inputText := flags.String("input", "", "optional original input context")
	targetOutput := flags.String("target-output", "", "existing target output to judge")
	groundTruth := flags.String("ground-truth", "", "optional expected answer or reference")
	rubric := flags.String("rubric", "", "judge rubric text")
	mode := flags.String("mode", string(sigmaevals.ModeEvaluate), "judge mode: evaluate or g_eval")
	passThreshold := flags.Float64("pass-threshold", 0, "G-Eval pass threshold on the 1-5 score scale; default 3")
	sessionID := flags.String("session-id", "", "optional Sigma session ID for provider affinity and prompt-cache keys")
	cacheRetention := flags.String("cache-retention", "", "optional provider prompt-cache retention: none, short, long, ephemeral, or persistent")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*judgeRaw) == "" {
		return fmt.Errorf("--judge is required")
	}
	if strings.TrimSpace(*targetOutput) == "" {
		return fmt.Errorf("--target-output is required")
	}
	judge, err := sigmaevals.ParseTarget(*judgeRaw)
	if err != nil {
		return err
	}
	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		return err
	}
	options, err := requestOptions(*sessionID, *cacheRetention)
	if err != nil {
		return err
	}
	client := sigma.NewClient(sigma.WithRegistry(registry))
	result, err := sigmaevals.NewTargetEvaluator(sigmaevals.SigmaTargetCompleter{Client: client, Registry: registry}).Judge(context.Background(), sigmaevals.JudgeInput{
		Input:         *inputText,
		TargetOutput:  *targetOutput,
		GroundTruth:   *groundTruth,
		Rubric:        *rubric,
		Judge:         judge,
		Mode:          sigmaevals.Mode(*mode),
		PassThreshold: *passThreshold,
		JudgeOptions:  options,
	})
	if err != nil {
		return err
	}
	if err := writeOutput(*outPath, *outFormat, result); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "judge-output: score=%.3f passed=%t mode=%s\n", result.Score, result.Passed, result.Mode)
	if !result.Passed {
		return errors.New("judge-output failed")
	}
	return nil
}

func runJudgeBatch(args []string) error {
	flags := flag.NewFlagSet("judge-batch", flag.ContinueOnError)
	name := flags.String("name", "batch-judge", "batch run name")
	casesPath := flags.String("cases", "", "JSON file containing an array of sigmaevals.JudgeCase")
	targetRaw := flags.String("target", "", "optional target as provider=model; required when cases omit targetOutput")
	judgeRaw := flags.String("judge", "", "judge target as provider=model")
	rubric := flags.String("rubric", "", "judge rubric ID or text")
	targetPrompt := flags.String("target-prompt", "", "optional prompt prepended to case inputs for target calls")
	mode := flags.String("mode", string(sigmaevals.ModeEvaluate), "judge mode: evaluate or g_eval")
	passThreshold := flags.Float64("pass-threshold", 0, "G-Eval pass threshold on the 1-5 score scale; default 3")
	repeats := flags.Int("repeat", 1, "number of repeats per judge case")
	concurrency := flags.Int("concurrency", 1, "number of concurrent judge cases")
	sessionID := flags.String("session-id", "", "optional Sigma session ID for provider affinity and prompt-cache keys")
	cacheRetention := flags.String("cache-retention", "", "optional provider prompt-cache retention")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	samplesOutPath := flags.String("samples-out", "", "optional variance samples JSONL output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*casesPath) == "" {
		return fmt.Errorf("--cases is required")
	}
	if strings.TrimSpace(*judgeRaw) == "" {
		return fmt.Errorf("--judge is required")
	}
	cases, err := readJudgeCases(*casesPath)
	if err != nil {
		return err
	}
	judge, err := sigmaevals.ParseTarget(*judgeRaw)
	if err != nil {
		return err
	}
	var target sigmaevals.Target
	if strings.TrimSpace(*targetRaw) != "" {
		target, err = sigmaevals.ParseTarget(*targetRaw)
		if err != nil {
			return err
		}
	}
	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		return err
	}
	options, err := requestOptions(*sessionID, *cacheRetention)
	if err != nil {
		return err
	}
	client := sigma.NewClient(sigma.WithRegistry(registry))
	result, err := sigmaevals.NewTargetEvaluator(sigmaevals.SigmaTargetCompleter{Client: client, Registry: registry}).EvaluateBatch(context.Background(), sigmaevals.BatchJudgeSpec{
		Name:          *name,
		Cases:         cases,
		Target:        target,
		Judge:         judge,
		Mode:          sigmaevals.Mode(*mode),
		Rubric:        *rubric,
		TargetPrompt:  *targetPrompt,
		PassThreshold: *passThreshold,
		TargetOptions: options,
		JudgeOptions:  options,
		Repeats:       *repeats,
		Concurrency:   *concurrency,
	})
	if err != nil {
		return err
	}
	if err := writeOutput(*outPath, *outFormat, result); err != nil {
		return err
	}
	if err := writeSamplesJSONL(*samplesOutPath, sigmaevals.VarianceSamplesFromBatchJudgeResult(result)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %d/%d passed, failed=%d, errors=%d\n", result.Name, result.Summary.Passed, result.Summary.Total, result.Summary.Failed, result.Summary.Errors)
	if result.Summary.Errors > 0 {
		return errors.New("batch judge recorded errors")
	}
	return nil
}

func runJudgeAlignment(args []string) error {
	flags := flag.NewFlagSet("judge-alignment", flag.ContinueOnError)
	name := flags.String("name", "judge-alignment", "judge alignment run name")
	casesPath := flags.String("cases", "", "JSON file containing an array of sigmaevals.JudgeAlignmentCase")
	judgeRaw := flags.String("judge", "", "judge target as provider=model")
	mode := flags.String("mode", string(sigmaevals.ModeEvaluate), "judge mode: evaluate or g_eval")
	passThreshold := flags.Float64("pass-threshold", 0, "G-Eval pass threshold on the 1-5 score scale; default 3")
	tolerance := flags.Float64("tolerance", 0.5, "score tolerance for alignment accuracy")
	concurrency := flags.Int("concurrency", 1, "number of concurrent judge cases")
	sessionID := flags.String("session-id", "", "optional Sigma session ID for provider affinity and prompt-cache keys")
	cacheRetention := flags.String("cache-retention", "", "optional provider prompt-cache retention")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	samplesOutPath := flags.String("samples-out", "", "optional variance samples JSONL output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*casesPath) == "" {
		return fmt.Errorf("--cases is required")
	}
	if strings.TrimSpace(*judgeRaw) == "" {
		return fmt.Errorf("--judge is required")
	}
	cases, err := readJudgeAlignmentCases(*casesPath)
	if err != nil {
		return err
	}
	judge, err := sigmaevals.ParseTarget(*judgeRaw)
	if err != nil {
		return err
	}
	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		return err
	}
	options, err := requestOptions(*sessionID, *cacheRetention)
	if err != nil {
		return err
	}
	client := sigma.NewClient(sigma.WithRegistry(registry))
	result, err := sigmaevals.NewTargetEvaluator(sigmaevals.SigmaTargetCompleter{Client: client, Registry: registry}).EvaluateJudges(context.Background(), sigmaevals.JudgeAlignmentSpec{
		Name:          *name,
		Cases:         cases,
		JudgeTargets:  []sigmaevals.Target{judge},
		Mode:          sigmaevals.Mode(*mode),
		PassThreshold: *passThreshold,
		Options:       options,
		Tolerance:     *tolerance,
		Concurrency:   *concurrency,
	})
	if err != nil {
		return err
	}
	if err := writeOutput(*outPath, *outFormat, result); err != nil {
		return err
	}
	if err := writeSamplesJSONL(*samplesOutPath, sigmaevals.VarianceSamplesFromJudgeAlignmentResult(result)); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: total=%d errors=%d tolerance_accuracy=%.3f classification_accuracy=%.3f\n", result.Name, result.Summary.Total, result.Summary.Errors, result.Summary.ToleranceAccuracy, result.Summary.Classification.Accuracy)
	if result.Summary.Errors > 0 {
		return errors.New("judge alignment recorded errors")
	}
	return nil
}

func runListRubrics(args []string) error {
	flags := flag.NewFlagSet("list-rubrics", flag.ContinueOnError)
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	if err := flags.Parse(args); err != nil {
		return err
	}
	return writeOutput(*outPath, *outFormat, sigmaevals.DefaultRubricRegistry.List())
}

func runVarianceReport(args []string) error {
	flags := flag.NewFlagSet("variance-report", flag.ContinueOnError)
	layer := flags.String("layer", "suite", "input layer: suite, batch-judge, judge-alignment, or samples")
	inputPath := flags.String("input", "", "input JSON result file, or JSONL samples when --layer samples")
	name := flags.String("name", "", "optional report name override")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	samplesOutPath := flags.String("samples-out", "", "optional normalized variance samples JSONL output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*inputPath) == "" {
		return fmt.Errorf("--input is required")
	}
	samples, defaultName, err := varianceSamplesFromFile(*layer, *inputPath)
	if err != nil {
		return err
	}
	reportName := firstNonEmpty(*name, defaultName)
	report := sigmaevals.BuildVarianceReport(reportName, samples)
	if err := writeOutput(*outPath, *outFormat, report); err != nil {
		return err
	}
	if err := writeSamplesJSONL(*samplesOutPath, samples); err != nil {
		return err
	}
	printVarianceReport(os.Stderr, report)
	return nil
}

func runCompareVariance(args []string) error {
	flags := flag.NewFlagSet("compare-variance", flag.ContinueOnError)
	baselinePath := flags.String("baseline", "", "baseline VarianceReport JSON file")
	currentPath := flags.String("current", "", "current VarianceReport JSON file")
	confidenceZ := flags.Float64("confidence-z", 1.96, "two-sided z threshold")
	minPassRateDelta := flags.Float64("min-pass-rate-delta", 0, "minimum absolute pass-rate delta for direction")
	minMeanScoreDelta := flags.Float64("min-mean-score-delta", 0, "minimum absolute mean-score delta for direction")
	outPath := flags.String("out", "", "optional output path; stdout when omitted")
	outFormat := flags.String("format", outputJSON, "output format: json, jsonl, or md")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*baselinePath) == "" || strings.TrimSpace(*currentPath) == "" {
		return fmt.Errorf("--baseline and --current are required")
	}
	var baseline sigmaevals.VarianceReport
	if err := readJSON(*baselinePath, &baseline); err != nil {
		return err
	}
	var current sigmaevals.VarianceReport
	if err := readJSON(*currentPath, &current); err != nil {
		return err
	}
	comparison := sigmaevals.CompareVarianceReports(baseline, current, sigmaevals.VarianceCompareOptions{
		ConfidenceZ:       *confidenceZ,
		MinPassRateDelta:  *minPassRateDelta,
		MinMeanScoreDelta: *minMeanScoreDelta,
	})
	if err := writeOutput(*outPath, *outFormat, comparison); err != nil {
		return err
	}
	printVarianceComparison(os.Stderr, comparison)
	if comparison.Summary.Regressions > 0 {
		return errors.New("variance comparison found regressions")
	}
	return nil
}

func registerCommonProviders(registry *sigma.Registry) error {
	registrations := []func() error{
		func() error { return anthropic.Register(registry, sigma.ProviderAnthropic) },
		func() error { return bedrock.Register(registry, sigma.ProviderAmazonBedrock) },
		func() error { return fireworks.Register(registry) },
		func() error { return google.Register(registry, sigma.ProviderGoogle) },
		func() error { return google.Register(registry, sigma.ProviderGoogleVertex) },
		func() error { return mistral.Register(registry, sigma.ProviderMistral) },
		func() error { return opencode.RegisterDefault(registry) },
		func() error { return openai.Register(registry, sigma.ProviderOpenAI) },
		func() error { return openai.RegisterImages(registry, sigma.ProviderOpenAI) },
		func() error { return openrouter.Register(registry) },
		func() error { return xai.Register(registry) },
	}
	for _, register := range registrations {
		if err := register(); err != nil {
			return err
		}
	}
	return nil
}

func requestOptions(sessionID string, cacheRetention string) ([]sigma.Option, error) {
	var options []sigma.Option
	if strings.TrimSpace(sessionID) != "" {
		options = append(options, sigma.WithSessionID(strings.TrimSpace(sessionID)))
	}
	if strings.TrimSpace(cacheRetention) == "" {
		return options, nil
	}
	retention, err := parseCacheRetention(cacheRetention)
	if err != nil {
		return nil, err
	}
	options = append(options, sigma.WithCacheRetention(retention))
	return options, nil
}

func parseCacheRetention(raw string) (sigma.CacheRetention, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "none":
		return sigma.CacheRetentionNone, nil
	case "short":
		return sigma.CacheRetentionShort, nil
	case "long":
		return sigma.CacheRetentionLong, nil
	case "ephemeral":
		return sigma.CacheRetentionEphemeral, nil
	case "persistent":
		return sigma.CacheRetentionPersistent, nil
	default:
		return "", fmt.Errorf("unknown cache retention %q", raw)
	}
}

type repeatedString []string

func (s *repeatedString) String() string { return strings.Join(*s, ",") }
func (s *repeatedString) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func exampleSuitePaths(root string) ([]string, error) {
	var paths []string
	if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || filepath.Ext(path) != ".json" {
			return nil
		}
		paths = append(paths, path)
		return nil
	}); err != nil {
		return nil, err
	}
	sort.Strings(paths)
	return paths, nil
}

type scriptedTargetCompleter struct {
	responses map[string]sigma.AssistantMessage
}

func scriptedCompleterForSuite(suite sigmaevals.Suite) (scriptedTargetCompleter, error) {
	responses := make(map[string]sigma.AssistantMessage, len(suite.Cases))
	renderer := sigmaevals.DefaultRenderer{}
	for _, c := range suite.Cases {
		request, err := renderer.Render(context.Background(), sigmaevals.RenderInput{Suite: suite, Case: c})
		if err != nil {
			return scriptedTargetCompleter{}, err
		}
		responses[requestKey(request)] = scriptedMessage(c)
	}
	return scriptedTargetCompleter{responses: responses}, nil
}

func (c scriptedTargetCompleter) CompleteTarget(_ context.Context, input sigmaevals.TargetRequest) (sigmaevals.TargetResult, error) {
	message, ok := c.responses[requestKey(input.Request)]
	if !ok {
		return sigmaevals.TargetResult{Target: input.Target, Request: input.Request, Repeat: input.Repeat, Error: "no scripted response"}, nil
	}
	result := sigmaevals.TargetResult{Target: input.Target, Request: input.Request, Repeat: input.Repeat, Message: message}
	if text, err := sigmaevals.AssistantText(message); err == nil {
		result.Output = text
	}
	return result, nil
}

func scriptedMessage(c sigmaevals.Case) sigma.AssistantMessage {
	if len(c.Expected.ToolCalls) > 0 {
		call := c.Expected.ToolCalls[0]
		return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.ToolCallBlock("call_1", call.Name, call.Arguments)}}
	}
	if c.Expected.JSON != nil {
		b, _ := json.Marshal(c.Expected.JSON)
		return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text(string(b))}}
	}
	if len(c.Expected.CorrectChoices) > 0 {
		return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text(c.Expected.CorrectChoices[0])}}
	}
	if len(c.Expected.Answers) > 0 {
		return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text(c.Expected.Answers[0])}}
	}
	if c.Expected.Output != "" {
		return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text(c.Expected.Output)}}
	}
	return sigma.AssistantMessage{Content: []sigma.ContentBlock{sigma.Text("ok")}}
}

func requestKey(request sigma.Request) string {
	b, _ := json.Marshal(request)
	return string(b)
}

func printVarianceReport(w io.Writer, report sigmaevals.VarianceReport) {
	fmt.Fprintf(w, "%s: groups=%d pass_rate=%.3f mean_score=%.3f stddev=%.3f stderr=%.3f unique_outputs=%d errors=%d\n", report.Name, len(report.Groups), report.Total.PassRate, report.Total.MeanScore, report.Total.StdDevScore, report.Total.StdErrScore, report.Total.UniqueOutputs, report.Total.Errors)
	for _, group := range report.Groups {
		fmt.Fprintf(w, "  %s model=%s scorer=%s n=%d pass_rate=%.3f mean=%.3f stddev=%.3f stderr=%.3f unique_outputs=%d errors=%d\n",
			group.Key.CaseID,
			valueOrDash(group.Key.Model),
			valueOrDash(group.Key.Scorer),
			group.Count,
			group.PassRate,
			group.MeanScore,
			group.StdDevScore,
			group.StdErrScore,
			group.UniqueOutputs,
			group.Errors,
		)
	}
}

func printVarianceComparison(w io.Writer, comparison sigmaevals.VarianceComparison) {
	fmt.Fprintf(w, "%s vs %s: regressions=%d improvements=%d stable=%d new=%d missing=%d\n", comparison.BaselineName, comparison.CurrentName, comparison.Summary.Regressions, comparison.Summary.Improvements, comparison.Summary.Stable, comparison.Summary.New, comparison.Summary.Missing)
	for _, delta := range comparison.Deltas {
		if delta.Direction == "stable" {
			continue
		}
		fmt.Fprintf(w, "  %s %s model=%s scorer=%s pass_delta=%.3f score_delta=%.3f score_z=%.3f pass_z=%.3f significant=%t\n",
			delta.Direction,
			delta.Key.CaseID,
			valueOrDash(delta.Key.Model),
			valueOrDash(delta.Key.Scorer),
			delta.DeltaPassRate,
			delta.DeltaMeanScore,
			delta.ScoreZ,
			delta.PassRateZ,
			delta.Significant,
		)
	}
}

func valueOrDash(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}

func readJudgeCases(path string) ([]sigmaevals.JudgeCase, error) {
	var cases []sigmaevals.JudgeCase
	if err := readJSON(path, &cases); err != nil {
		return nil, err
	}
	return cases, nil
}

func readJudgeAlignmentCases(path string) ([]sigmaevals.JudgeAlignmentCase, error) {
	var cases []sigmaevals.JudgeAlignmentCase
	if err := readJSON(path, &cases); err != nil {
		return nil, err
	}
	return cases, nil
}

func varianceSamplesFromFile(layer string, path string) ([]sigmaevals.VarianceSample, string, error) {
	switch strings.ToLower(strings.TrimSpace(layer)) {
	case "samples", "sample", "jsonl":
		file, err := os.Open(path)
		if err != nil {
			return nil, "", err
		}
		defer file.Close()
		samples, err := sigmaevals.ReadVarianceSamplesJSONL(file)
		return samples, filepath.Base(path), err
	case "suite", "run", "run-result":
		var run sigmaevals.RunResult
		if err := readJSON(path, &run); err != nil {
			return nil, "", err
		}
		return sigmaevals.VarianceSamplesFromRunResult(run), run.SuiteName, nil
	case "batch", "batch-judge", "batch_judge", "judge-batch":
		var run sigmaevals.BatchJudgeResult
		if err := readJSON(path, &run); err != nil {
			return nil, "", err
		}
		return sigmaevals.VarianceSamplesFromBatchJudgeResult(run), run.Name, nil
	case "judge-alignment", "judge_alignment", "alignment":
		var run sigmaevals.JudgeAlignmentRunResult
		if err := readJSON(path, &run); err != nil {
			return nil, "", err
		}
		return sigmaevals.VarianceSamplesFromJudgeAlignmentResult(run), run.Name, nil
	default:
		return nil, "", fmt.Errorf("unknown variance layer %q", layer)
	}
}

func readJSON(path string, value any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}

func writeSamplesJSONL(path string, samples []sigmaevals.VarianceSample) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return sigmaevals.WriteVarianceSamplesJSONL(file, samples)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func writeJSON(path string, value any) error {
	return writeOutput(path, outputJSON, value)
}

func writeOutput(path string, format string, value any) error {
	if strings.TrimSpace(path) == "" {
		return writeRenderedOutput(os.Stdout, format, value)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return writeRenderedOutput(file, format, value)
}
