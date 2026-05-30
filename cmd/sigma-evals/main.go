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
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/fireworks"
	"github.com/wintermi/sigma/provider/openai"
	"github.com/wintermi/sigma/provider/opencode"
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
  smoke-examples  Run all JSON examples with a scripted TargetCompleter, no network.
  run-suite       Run one suite against real Sigma targets.

Examples:
  sigma-evals smoke-examples --examples examples --out runs/smoke.json
  sigma-evals run-suite --suite examples/generic/answer-aliases.json --target fireworks=accounts/fireworks/routers/kimi-k2p6-turbo`)
}

func runSmokeExamples(args []string) error {
	flags := flag.NewFlagSet("smoke-examples", flag.ContinueOnError)
	examplesDir := flags.String("examples", "examples", "directory containing example JSON suites")
	outPath := flags.String("out", "", "optional JSON output path")
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
	if err := writeJSON(*outPath, summary); err != nil {
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
	outPath := flags.String("out", "", "optional JSON output path")
	repeats := flags.Int("repeat", 1, "number of repeats per target")
	concurrency := flags.Int("concurrency", 1, "number of concurrent target attempts")
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
	client := sigma.NewClient(sigma.WithRegistry(registry))
	result, err := sigmaevals.NewTargetRunner(sigmaevals.SigmaTargetCompleter{Client: client, Registry: registry}).Run(context.Background(), sigmaevals.TargetRunSpec{
		Suite:       suite,
		Targets:     targets,
		Scorers:     []sigmaevals.Scorer{sigmaevals.AutoScorer{}},
		Repeats:     *repeats,
		Concurrency: *concurrency,
	})
	if err != nil {
		return err
	}
	if err := writeJSON(*outPath, result); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %d/%d passed, failed=%d, errors=%d\n", result.SuiteName, result.Summary.Passed, result.Summary.Total, result.Summary.Failed, result.Summary.Errors)
	if result.Summary.Failed > 0 || result.Summary.Errors > 0 {
		return errors.New("suite run failed")
	}
	return nil
}

func registerCommonProviders(registry *sigma.Registry) error {
	if err := fireworks.Register(registry); err != nil {
		return err
	}
	if err := opencode.RegisterDefault(registry); err != nil {
		return err
	}
	return openai.Register(registry, sigma.ProviderOpenAI)
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

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if strings.TrimSpace(path) == "" {
		fmt.Println(string(data))
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
