// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/wintermi/sigma"
)

// Runner executes suites through Sigma models.
type Runner struct {
	Client Completer
}

// NewRunner constructs a Runner. A nil client uses sigma.NewClient at run time.
func NewRunner(client Completer) *Runner {
	return &Runner{Client: client}
}

// Run executes the suite across all configured models and repeats.
func (r *Runner) Run(ctx context.Context, spec RunSpec) (RunResult, error) {
	if strings.TrimSpace(spec.Suite.Name) == "" {
		return RunResult{}, fmt.Errorf("suite name is required")
	}
	if len(spec.Suite.Cases) == 0 {
		return RunResult{}, fmt.Errorf("suite must contain at least one case")
	}
	if len(spec.Models) == 0 {
		return RunResult{}, fmt.Errorf("at least one model is required")
	}

	repeats := spec.Repeats
	if repeats <= 0 {
		repeats = 1
	}
	concurrency := spec.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	renderer := rendererOrDefault(spec.Renderer)
	scorers := scorersOrDefault(spec.Scorers)
	client := r.clientOrDefault()

	startedAt := time.Now().UTC()
	jobs := make(chan runJob)
	var mu sync.Mutex
	results := make([]CaseResult, 0, len(spec.Suite.Cases)*len(spec.Models)*repeats)

	worker := func() {
		for job := range jobs {
			result := r.runOne(ctx, client, renderer, scorers, spec, job)
			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}
	}

	workerCount := concurrency
	totalJobs := len(spec.Suite.Cases) * len(spec.Models) * repeats
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

	for _, c := range spec.Suite.Cases {
		for _, model := range spec.Models {
			for repeat := 1; repeat <= repeats; repeat++ {
				select {
				case <-ctx.Done():
					close(jobs)
					wg.Wait()
					return RunResult{}, ctx.Err()
				case jobs <- runJob{Case: c, Model: model, Repeat: repeat}:
				}
			}
		}
	}
	close(jobs)
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].CaseID != results[j].CaseID {
			return results[i].CaseID < results[j].CaseID
		}
		if results[i].Model != results[j].Model {
			return results[i].Model < results[j].Model
		}
		return results[i].Repeat < results[j].Repeat
	})

	endedAt := time.Now().UTC()
	summary, byModel, byTag := summarize(results)
	return RunResult{
		SuiteName:    spec.Suite.Name,
		SuiteVersion: spec.Suite.Version,
		StartedAt:    startedAt,
		EndedAt:      endedAt,
		Results:      results,
		Summary:      summary,
		ByModel:      byModel,
		ByTag:        byTag,
	}, nil
}

type runJob struct {
	Case   Case
	Model  sigma.Model
	Repeat int
}

func (r *Runner) runOne(ctx context.Context, client Completer, renderer Renderer, scorers []Scorer, spec RunSpec, job runJob) CaseResult {
	startedAt := time.Now()
	result := CaseResult{
		CaseID:   job.Case.ID,
		CaseName: job.Case.Name,
		Tags:     append([]string(nil), job.Case.Tags...),
		Model:    modelLabel(job.Model),
		Provider: job.Model.Provider,
		Repeat:   job.Repeat,
	}

	request, err := renderer.Render(ctx, RenderInput{Suite: spec.Suite, Case: job.Case})
	result.Request = request
	if err != nil {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
		result.DurationMS = time.Since(startedAt).Milliseconds()
		return result
	}

	message, err := client.Complete(ctx, job.Model, request, spec.Options...)
	result.Message = message
	result.Usage = message.Usage
	result.Cost = message.Cost
	result.ProviderMeta = message.ProviderMetadata
	if err != nil {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
		result.DurationMS = time.Since(startedAt).Milliseconds()
		return result
	}

	output, err := AssistantText(message)
	result.Output = output
	if err != nil && len(job.Case.Expected.ToolCalls) == 0 {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
		result.DurationMS = time.Since(startedAt).Milliseconds()
		return result
	}

	for _, scorer := range scorers {
		score, err := scorer.Score(ctx, ScoreInput{
			Suite:   spec.Suite,
			Case:    job.Case,
			Model:   job.Model,
			Repeat:  job.Repeat,
			Request: request,
			Output:  output,
			Message: message,
		})
		if err != nil {
			result.Scores = append(result.Scores, Score{
				Name:      scorer.Name(),
				Score:     0,
				Passed:    false,
				Rationale: err.Error(),
			})
			continue
		}
		if score.Name == "" {
			score.Name = scorer.Name()
		}
		result.Scores = append(result.Scores, score)
	}
	result.DurationMS = time.Since(startedAt).Milliseconds()
	return result
}

func (r *Runner) clientOrDefault() Completer {
	if r == nil || r.Client == nil {
		return sigma.NewClient()
	}
	return r.Client
}

func summarize(results []CaseResult) (Summary, map[string]ModelSummary, map[string]Summary) {
	var summary Summary
	byModel := make(map[string]ModelSummary)
	byTag := make(map[string]Summary)
	for _, result := range results {
		passed := casePassed(result)
		summary = addSummaryResult(summary, result, passed)

		modelSummary := byModel[result.Model]
		modelSummary.Total++
		modelSummary.DurationMS += result.DurationMS
		if result.Error != "" {
			modelSummary.Errors++
		} else {
			for _, score := range result.Scores {
				modelSummary.ScoreCount++
				modelSummary.MeanScore += score.Score
			}
			if passed {
				modelSummary.Passed++
			} else {
				modelSummary.Failed++
			}
		}
		byModel[result.Model] = modelSummary

		for _, tag := range result.Tags {
			byTag[tag] = addSummaryResult(byTag[tag], result, passed)
		}
	}
	for model, modelSummary := range byModel {
		if modelSummary.ScoreCount > 0 {
			modelSummary.MeanScore /= float64(modelSummary.ScoreCount)
		}
		byModel[model] = modelSummary
	}
	return summary, byModel, byTag
}

func casePassed(result CaseResult) bool {
	if result.Error != "" {
		return false
	}
	passed := len(result.Scores) > 0
	for _, score := range result.Scores {
		if !score.Passed {
			passed = false
		}
	}
	return passed
}

func addSummaryResult(summary Summary, result CaseResult, passed bool) Summary {
	summary.Total++
	if result.Error != "" {
		summary.Errors++
		return summary
	}
	for range result.Scores {
		summary.ScoreCount++
	}
	if passed {
		summary.Passed++
	} else {
		summary.Failed++
	}
	return summary
}

func modelLabel(model sigma.Model) string {
	if model.Name != "" {
		return model.Name
	}
	if model.ID != "" {
		return string(model.ID)
	}
	return string(model.Provider)
}
