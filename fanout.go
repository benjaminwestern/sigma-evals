// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/wintermi/sigma"
)

// ProgressKind identifies runner progress event types.
type ProgressKind string

const (
	// ProgressStart means a case or target attempt was queued for execution.
	ProgressStart ProgressKind = "start"
	// ProgressResult means a case or target attempt completed.
	ProgressResult ProgressKind = "result"
)

// ProgressEvent is a small SDK progress notification. It is intentionally
// synchronous and side-effect free: callers decide whether to log, stream, save,
// or ignore it.
type ProgressEvent struct {
	Kind         ProgressKind  `json:"kind"`
	CaseID       string        `json:"caseId,omitempty"`
	Target       Target        `json:"target,omitempty"`
	Repeat       int           `json:"repeat,omitempty"`
	Result       *CaseResult   `json:"result,omitempty"`
	TargetResult *TargetResult `json:"targetResult,omitempty"`
}

// ProgressFunc receives runner progress events.
type ProgressFunc func(ProgressEvent)

// FanoutSpec configures a raw request fanout across targets. It does not score
// outputs; use TargetRunner for suite/case scoring.
type FanoutSpec struct {
	Request     sigma.Request  `json:"request"`
	Targets     []Target       `json:"targets"`
	Options     []sigma.Option `json:"-"`
	Repeats     int            `json:"repeats,omitempty"`
	Concurrency int            `json:"concurrency,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Progress    ProgressFunc   `json:"-"`
}

// FanoutResult is the portable output of a raw target fanout.
type FanoutResult struct {
	StartedAt time.Time      `json:"startedAt"`
	EndedAt   time.Time      `json:"endedAt"`
	Request   sigma.Request  `json:"request"`
	Results   []TargetResult `json:"results"`
	Summary   FanoutSummary  `json:"summary"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// FanoutSummary aggregates raw target attempts.
type FanoutSummary struct {
	Total      int   `json:"total"`
	Succeeded  int   `json:"succeeded"`
	Failed     int   `json:"failed"`
	DurationMS int64 `json:"durationMs"`
}

// RunFanout executes one rendered request against all targets and repeats.
func RunFanout(ctx context.Context, completer TargetCompleter, spec FanoutSpec) (FanoutResult, error) {
	if len(spec.Request.Messages) == 0 && spec.Request.SystemPrompt == "" {
		return FanoutResult{}, fmt.Errorf("request must contain messages or a system prompt")
	}
	if len(spec.Targets) == 0 {
		return FanoutResult{}, fmt.Errorf("at least one target is required")
	}
	if completer == nil {
		completer = SigmaTargetCompleter{}
	}
	repeats := repeatOrOne(spec.Repeats)
	concurrency := concurrencyOrOne(spec.Concurrency)
	startedAt := time.Now().UTC()
	jobs := make(chan fanoutJob)
	var mu sync.Mutex
	results := make([]TargetResult, 0, len(spec.Targets)*repeats)

	worker := func() {
		for job := range jobs {
			if spec.Progress != nil {
				spec.Progress(ProgressEvent{Kind: ProgressStart, Target: job.Target, Repeat: job.Repeat})
			}
			result, err := completer.CompleteTarget(ctx, TargetRequest{
				Target:   job.Target,
				Request:  spec.Request,
				Options:  spec.Options,
				Repeat:   job.Repeat,
				Metadata: spec.Metadata,
			})
			if err != nil && result.Error == "" {
				result.Error = err.Error()
			}
			if result.ErrorDetails == nil {
				result.ErrorDetails = classifyErrorOrMessage(err, result.Error)
			}
			if spec.Progress != nil {
				copyResult := result
				spec.Progress(ProgressEvent{Kind: ProgressResult, Target: job.Target, Repeat: job.Repeat, TargetResult: &copyResult})
			}
			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}
	}

	workerCount := min(concurrency, len(spec.Targets)*repeats)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			worker()
		}()
	}
	for _, target := range spec.Targets {
		for repeat := 1; repeat <= repeats; repeat++ {
			select {
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return FanoutResult{}, ctx.Err()
			case jobs <- fanoutJob{Target: target, Repeat: repeat}:
			}
		}
	}
	close(jobs)
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		left := results[i].Target.labelOrDefault()
		right := results[j].Target.labelOrDefault()
		if left != right {
			return left < right
		}
		return results[i].Repeat < results[j].Repeat
	})
	endedAt := time.Now().UTC()
	return FanoutResult{
		StartedAt: startedAt,
		EndedAt:   endedAt,
		Request:   spec.Request,
		Results:   results,
		Summary:   summarizeFanout(results, endedAt.Sub(startedAt).Milliseconds()),
		Metadata:  copyMap(spec.Metadata),
	}, nil
}

type fanoutJob struct {
	Target Target
	Repeat int
}

func summarizeFanout(results []TargetResult, durationMS int64) FanoutSummary {
	summary := FanoutSummary{Total: len(results), DurationMS: durationMS}
	for _, result := range results {
		if result.Error != "" {
			summary.Failed++
		} else {
			summary.Succeeded++
		}
	}
	return summary
}
