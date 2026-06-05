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

// Target identifies one model/runtime endpoint a caller wants to evaluate.
// Apps may attach their own routing metadata without making sigma-evals own
// their session, persistence, or provider lifecycle.
type Target struct {
	Provider    sigma.ProviderID `json:"provider"`
	ModelID     sigma.ModelID    `json:"model"`
	Label       string           `json:"label,omitempty"`
	Name        string           `json:"name,omitempty"`
	ModelConfig *sigma.Model     `json:"modelConfig,omitempty"`
	Options     map[string]any   `json:"options,omitempty"`
	Metadata    map[string]any   `json:"metadata,omitempty"`
}

// TargetRequest is one rendered request for one target attempt.
type TargetRequest struct {
	Target   Target         `json:"target"`
	Request  sigma.Request  `json:"request"`
	Options  []sigma.Option `json:"-"`
	Repeat   int            `json:"repeat,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// TargetResult records the raw model result for one target attempt, before or
// after scoring.
type TargetResult struct {
	Target           Target                 `json:"target"`
	Request          sigma.Request          `json:"request,omitempty"`
	Repeat           int                    `json:"repeat,omitempty"`
	Output           string                 `json:"output,omitempty"`
	Message          sigma.AssistantMessage `json:"message,omitempty"`
	Error            string                 `json:"error,omitempty"`
	ErrorDetails     *ErrorDetails          `json:"errorDetails,omitempty"`
	DurationMS       int64                  `json:"durationMs"`
	Usage            *sigma.Usage           `json:"usage,omitempty"`
	Cost             *sigma.Cost            `json:"cost,omitempty"`
	ProviderMetadata map[string]any         `json:"providerMetadata,omitempty"`
	Logprobs         []TokenLogprob         `json:"logprobs,omitempty"`
	Metadata         map[string]any         `json:"metadata,omitempty"`
}

// TargetCompleter is the SDK boundary for running model calls. Direct Sigma,
// agentic-control sessions, local models, hosted apps, or tests can implement
// this without sigma-evals depending on their app machinery.
type TargetCompleter interface {
	CompleteTarget(context.Context, TargetRequest) (TargetResult, error)
}

// SigmaTargetCompleter adapts a Sigma client to TargetCompleter.
type SigmaTargetCompleter struct {
	Client   Completer
	Registry *sigma.Registry
}

// NewSigmaTargetCompleter constructs a TargetCompleter backed by Sigma.
func NewSigmaTargetCompleter(client Completer) SigmaTargetCompleter {
	return SigmaTargetCompleter{Client: client}
}

// CompleteTarget implements TargetCompleter.
func (c SigmaTargetCompleter) CompleteTarget(ctx context.Context, input TargetRequest) (TargetResult, error) {
	started := time.Now()
	result := TargetResult{Target: input.Target, Request: input.Request, Repeat: repeatOrOne(input.Repeat), Metadata: copyMap(input.Metadata)}
	model, err := c.model(input.Target)
	if err != nil {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
		result.DurationMS = time.Since(started).Milliseconds()
		return result, err
	}
	client := c.Client
	if client == nil {
		client = sigma.NewClient(sigma.WithRegistry(c.registry()))
	}
	message, err := client.Complete(ctx, model, input.Request, input.Options...)
	result.Message = message
	result.Usage = message.Usage
	result.Cost = message.Cost
	result.ProviderMetadata = message.ProviderMetadata
	if logprobs, ok := TokenLogprobsFromMetadata(message.ProviderMetadata); ok {
		result.Logprobs = logprobs
	}
	if err != nil {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
		result.DurationMS = time.Since(started).Milliseconds()
		return result, err
	}
	if text, err := AssistantText(message); err == nil {
		result.Output = text
	} else if len(AssistantToolCalls(message)) == 0 {
		result.Error = err.Error()
		result.ErrorDetails = classifyError(err)
	}
	result.DurationMS = time.Since(started).Milliseconds()
	return result, nil
}

func (c SigmaTargetCompleter) model(target Target) (sigma.Model, error) {
	if target.ModelConfig != nil {
		model := *target.ModelConfig
		if model.Provider == "" {
			model.Provider = target.Provider
		}
		if model.ID == "" {
			model.ID = target.ModelID
		}
		return model, nil
	}
	if target.Provider == "" || target.ModelID == "" {
		return sigma.Model{}, fmt.Errorf("target provider and model are required")
	}
	if model, ok := c.registry().Model(target.Provider, target.ModelID); ok {
		return model, nil
	}
	return sigma.Model{Provider: target.Provider, ID: target.ModelID, Name: target.Name}, nil
}

func (c SigmaTargetCompleter) registry() *sigma.Registry {
	if c.Registry != nil {
		return c.Registry
	}
	return sigma.DefaultRegistry()
}

// TargetFromModel converts a Sigma model into a portable target.
func TargetFromModel(model sigma.Model) Target {
	modelCopy := model
	return Target{
		Provider:    model.Provider,
		ModelID:     model.ID,
		Name:        model.Name,
		ModelConfig: &modelCopy,
	}
}

// ParseTarget parses provider=model, provider/model, or provider:model target strings.
func ParseTarget(raw string) (Target, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Target{}, fmt.Errorf("target is required")
	}
	for _, sep := range []string{"=", ":", "/"} {
		parts := strings.SplitN(raw, sep, 2)
		if len(parts) == 2 && strings.TrimSpace(parts[0]) != "" && strings.TrimSpace(parts[1]) != "" {
			return Target{Provider: sigma.ProviderID(strings.TrimSpace(parts[0])), ModelID: sigma.ModelID(strings.TrimSpace(parts[1]))}, nil
		}
	}
	return Target{}, fmt.Errorf("target %q must be provider=model, provider/model, or provider:model", raw)
}

// TargetRunSpec configures a suite run across portable targets.
type TargetRunSpec struct {
	Suite       Suite          `json:"suite"`
	Targets     []Target       `json:"targets"`
	Renderer    Renderer       `json:"-"`
	Scorers     []Scorer       `json:"-"`
	Options     []sigma.Option `json:"-"`
	Repeats     int            `json:"repeats,omitempty"`
	Concurrency int            `json:"concurrency,omitempty"`
	Progress    ProgressFunc   `json:"-"`
}

// TargetRunner executes suites through a TargetCompleter.
type TargetRunner struct {
	Completer TargetCompleter
}

// NewTargetRunner constructs a TargetRunner.
func NewTargetRunner(completer TargetCompleter) *TargetRunner {
	return &TargetRunner{Completer: completer}
}

// Run executes the suite across all configured targets and repeats.
func (r *TargetRunner) Run(ctx context.Context, spec TargetRunSpec) (RunResult, error) {
	if strings.TrimSpace(spec.Suite.Name) == "" {
		return RunResult{}, fmt.Errorf("suite name is required")
	}
	if len(spec.Suite.Cases) == 0 {
		return RunResult{}, fmt.Errorf("suite must contain at least one case")
	}
	if len(spec.Targets) == 0 {
		return RunResult{}, fmt.Errorf("at least one target is required")
	}
	completer := r.completerOrDefault()
	repeats := repeatOrOne(spec.Repeats)
	concurrency := concurrencyOrOne(spec.Concurrency)
	renderer := rendererOrDefault(spec.Renderer)
	scorers := scorersOrDefault(spec.Scorers)

	startedAt := time.Now().UTC()
	jobs := make(chan targetRunJob)
	var mu sync.Mutex
	results := make([]CaseResult, 0, len(spec.Suite.Cases)*len(spec.Targets)*repeats)

	worker := func() {
		for job := range jobs {
			if spec.Progress != nil {
				spec.Progress(ProgressEvent{Kind: ProgressStart, CaseID: job.Case.ID, Target: job.Target, Repeat: job.Repeat})
			}
			result := runTargetCase(ctx, completer, renderer, scorers, spec, job)
			if spec.Progress != nil {
				spec.Progress(ProgressEvent{Kind: ProgressResult, CaseID: job.Case.ID, Target: job.Target, Repeat: job.Repeat, Result: &result})
			}
			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}
	}

	workerCount := min(concurrency, len(spec.Suite.Cases)*len(spec.Targets)*repeats)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	for _, c := range spec.Suite.Cases {
		for _, target := range spec.Targets {
			for repeat := 1; repeat <= repeats; repeat++ {
				select {
				case <-ctx.Done():
					close(jobs)
					wg.Wait()
					return RunResult{}, ctx.Err()
				case jobs <- targetRunJob{Case: c, Target: target, Repeat: repeat}:
				}
			}
		}
	}
	close(jobs)
	wg.Wait()

	sortCaseResults(results)
	summary, byModel, byTag := summarize(results)
	return RunResult{
		SuiteName:    spec.Suite.Name,
		SuiteVersion: spec.Suite.Version,
		StartedAt:    startedAt,
		EndedAt:      time.Now().UTC(),
		Results:      results,
		Summary:      summary,
		ByModel:      byModel,
		ByTag:        byTag,
	}, nil
}

func (r *TargetRunner) completerOrDefault() TargetCompleter {
	if r == nil || r.Completer == nil {
		return SigmaTargetCompleter{}
	}
	return r.Completer
}

type targetRunJob struct {
	Case   Case
	Target Target
	Repeat int
}

func runTargetCase(ctx context.Context, completer TargetCompleter, renderer Renderer, scorers []Scorer, spec TargetRunSpec, job targetRunJob) CaseResult {
	startedAt := time.Now()
	model := job.Target.modelForScoring()
	result := CaseResult{
		CaseID:   job.Case.ID,
		CaseName: job.Case.Name,
		Tags:     append([]string(nil), job.Case.Tags...),
		Model:    job.Target.labelOrDefault(),
		Provider: job.Target.Provider,
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

	targetResult, err := completer.CompleteTarget(ctx, TargetRequest{
		Target:  job.Target,
		Request: request,
		Options: spec.Options,
		Repeat:  job.Repeat,
		Metadata: map[string]any{
			"suite":  spec.Suite.Name,
			"caseId": job.Case.ID,
		},
	})
	result.Message = targetResult.Message
	result.Output = targetResult.Output
	result.Usage = targetResult.Usage
	result.Cost = targetResult.Cost
	result.ProviderMeta = targetResult.ProviderMetadata
	result.ErrorDetails = targetResult.ErrorDetails
	if targetResult.DurationMS > 0 {
		result.DurationMS = targetResult.DurationMS
	}
	if err != nil || targetResult.Error != "" {
		if targetResult.Error != "" {
			result.Error = targetResult.Error
		} else {
			result.Error = err.Error()
		}
		if result.ErrorDetails == nil {
			result.ErrorDetails = classifyErrorOrMessage(err, result.Error)
		}
		if result.DurationMS == 0 {
			result.DurationMS = time.Since(startedAt).Milliseconds()
		}
		return result
	}
	if result.Output == "" && len(result.Message.Content) > 0 {
		output, err := AssistantText(result.Message)
		result.Output = output
		if err != nil && len(job.Case.Expected.ToolCalls) == 0 {
			result.Error = err.Error()
			result.ErrorDetails = classifyError(err)
			if result.DurationMS == 0 {
				result.DurationMS = time.Since(startedAt).Milliseconds()
			}
			return result
		}
	}

	for _, scorer := range scorers {
		score, err := scorer.Score(ctx, ScoreInput{
			Suite:   spec.Suite,
			Case:    job.Case,
			Model:   model,
			Repeat:  job.Repeat,
			Request: request,
			Output:  result.Output,
			Message: result.Message,
		})
		if err != nil {
			result.Scores = append(result.Scores, Score{Name: scorer.Name(), Score: 0, Passed: false, Rationale: err.Error()})
			continue
		}
		if score.Name == "" {
			score.Name = scorer.Name()
		}
		result.Scores = append(result.Scores, score)
	}
	if result.DurationMS == 0 {
		result.DurationMS = time.Since(startedAt).Milliseconds()
	}
	return result
}

func (t Target) labelOrDefault() string {
	if strings.TrimSpace(t.Label) != "" {
		return strings.TrimSpace(t.Label)
	}
	if strings.TrimSpace(t.Name) != "" {
		return strings.TrimSpace(t.Name)
	}
	if t.Provider != "" && t.ModelID != "" {
		return string(t.Provider) + "/" + string(t.ModelID)
	}
	if t.Provider != "" {
		return string(t.Provider)
	}
	return string(t.ModelID)
}

func (t Target) modelForScoring() sigma.Model {
	if t.ModelConfig != nil {
		model := *t.ModelConfig
		if model.Provider == "" {
			model.Provider = t.Provider
		}
		if model.ID == "" {
			model.ID = t.ModelID
		}
		if model.Name == "" {
			model.Name = t.Name
		}
		return model
	}
	return sigma.Model{Provider: t.Provider, ID: t.ModelID, Name: t.Name}
}

func sortCaseResults(results []CaseResult) {
	sort.Slice(results, func(i, j int) bool {
		if results[i].CaseID != results[j].CaseID {
			return results[i].CaseID < results[j].CaseID
		}
		if results[i].Model != results[j].Model {
			return results[i].Model < results[j].Model
		}
		return results[i].Repeat < results[j].Repeat
	})
}

func repeatOrOne(value int) int {
	if value <= 0 {
		return 1
	}
	return value
}

func concurrencyOrOne(value int) int {
	if value <= 0 {
		return 1
	}
	return value
}
