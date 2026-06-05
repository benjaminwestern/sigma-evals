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

// JudgeCase is one item in a batch judge run. When TargetOutput is set, the
// batch runner judges that saved output directly. Otherwise it first runs the
// configured target using Input and TargetPrompt.
type JudgeCase struct {
	ID           string         `json:"id"`
	Name         string         `json:"name,omitempty"`
	Input        string         `json:"input,omitempty"`
	GroundTruth  string         `json:"groundTruth,omitempty"`
	TargetOutput string         `json:"targetOutput,omitempty"`
	Rubric       string         `json:"rubric,omitempty"`
	TargetPrompt string         `json:"targetPrompt,omitempty"`
	Tags         []string       `json:"tags,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// BatchJudgeSpec configures a batch of target-generation plus judge-evaluation
// items. It is intentionally host-neutral: callers provide a TargetCompleter
// through Evaluator and own persistence outside this package.
type BatchJudgeSpec struct {
	Name          string         `json:"name"`
	Version       string         `json:"version,omitempty"`
	Cases         []JudgeCase    `json:"cases"`
	Target        Target         `json:"target,omitempty"`
	Judge         Target         `json:"judge,omitempty"`
	TargetModel   sigma.Model    `json:"targetModel"`
	JudgeModel    sigma.Model    `json:"judgeModel"`
	Mode          Mode           `json:"mode,omitempty"`
	Rubric        string         `json:"rubric,omitempty"`
	TargetPrompt  string         `json:"targetPrompt,omitempty"`
	PassThreshold float64        `json:"passThreshold,omitempty"`
	TargetOptions []sigma.Option `json:"-"`
	JudgeOptions  []sigma.Option `json:"-"`
	Repeats       int            `json:"repeats,omitempty"`
	Concurrency   int            `json:"concurrency,omitempty"`
}

// BatchJudgeResult records a complete batch judge run.
type BatchJudgeResult struct {
	Name      string                       `json:"name"`
	Version   string                       `json:"version,omitempty"`
	StartedAt time.Time                    `json:"startedAt"`
	EndedAt   time.Time                    `json:"endedAt"`
	Results   []JudgeCaseResult            `json:"results"`
	Summary   BatchJudgeSummary            `json:"summary"`
	ByTag     map[string]BatchJudgeSummary `json:"byTag,omitempty"`
	ByModel   map[string]BatchJudgeSummary `json:"byModel,omitempty"`
	Metadata  map[string]any               `json:"metadata,omitempty"`
}

// JudgeCaseResult records one batch case result.
type JudgeCaseResult struct {
	CaseID         string           `json:"caseId"`
	CaseName       string           `json:"caseName,omitempty"`
	Tags           []string         `json:"tags,omitempty"`
	Target         string           `json:"target,omitempty"`
	TargetProvider sigma.ProviderID `json:"targetProvider,omitempty"`
	Judge          string           `json:"judge,omitempty"`
	JudgeProvider  sigma.ProviderID `json:"judgeProvider,omitempty"`
	Mode           Mode             `json:"mode"`
	Repeat         int              `json:"repeat,omitempty"`
	Score          float64          `json:"score,omitempty"`
	Rationale      string           `json:"rationale,omitempty"`
	Passed         bool             `json:"passed"`
	PassThreshold  float64          `json:"passThreshold,omitempty"`
	JSON           string           `json:"json,omitempty"`
	RawJudgeOutput string           `json:"rawJudgeOutput,omitempty"`
	TargetOutput   string           `json:"targetOutput,omitempty"`
	Result         *JudgeResult     `json:"result,omitempty"`
	Error          string           `json:"error,omitempty"`
	DurationMS     int64            `json:"durationMs"`
	Metadata       map[string]any   `json:"metadata,omitempty"`
}

// BatchJudgeSummary aggregates batch judge results.
type BatchJudgeSummary struct {
	Total      int     `json:"total"`
	Passed     int     `json:"passed"`
	Failed     int     `json:"failed"`
	Errors     int     `json:"errors"`
	MeanScore  float64 `json:"meanScore,omitempty"`
	ScoreCount int     `json:"scoreCount"`
	DurationMS int64   `json:"durationMs"`
}

// EvaluateBatch runs a batch of cases through Evaluate or Judge and returns
// stable, serializable result records. Per-case target or judge errors are
// recorded in Results; invalid specs and context cancellation are returned.
func (e *Evaluator) EvaluateBatch(ctx context.Context, spec BatchJudgeSpec) (BatchJudgeResult, error) {
	if strings.TrimSpace(spec.Name) == "" {
		return BatchJudgeResult{}, fmt.Errorf("batch judge name is required")
	}
	if len(spec.Cases) == 0 {
		return BatchJudgeResult{}, fmt.Errorf("batch judge spec must contain at least one case")
	}
	if isZeroTarget(targetWithModelFallback(spec.Judge, spec.JudgeModel)) {
		return BatchJudgeResult{}, fmt.Errorf("judge target is required")
	}
	seenCaseIDs := make(map[string]struct{}, len(spec.Cases))
	for _, c := range spec.Cases {
		caseID := strings.TrimSpace(c.ID)
		if caseID == "" {
			return BatchJudgeResult{}, fmt.Errorf("batch judge case id is required")
		}
		if _, ok := seenCaseIDs[caseID]; ok {
			return BatchJudgeResult{}, fmt.Errorf("duplicate batch judge case id %q", caseID)
		}
		seenCaseIDs[caseID] = struct{}{}
		if strings.TrimSpace(c.TargetOutput) == "" && isZeroTarget(targetWithModelFallback(spec.Target, spec.TargetModel)) {
			return BatchJudgeResult{}, fmt.Errorf("target is required for batch judge case %q without targetOutput", c.ID)
		}
	}

	repeats := repeatOrOne(spec.Repeats)
	concurrency := concurrencyOrOne(spec.Concurrency)
	startedAt := time.Now().UTC()
	jobs := make(chan batchJudgeJob)
	results := make([]JudgeCaseResult, 0, len(spec.Cases)*repeats)
	var mu sync.Mutex

	worker := func() {
		for job := range jobs {
			result := e.evaluateBatchCase(ctx, spec, job)
			mu.Lock()
			results = append(results, result)
			mu.Unlock()
		}
	}

	workerCount := min(concurrency, len(spec.Cases)*repeats)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			worker()
		}()
	}

	for _, c := range spec.Cases {
		for repeat := 1; repeat <= repeats; repeat++ {
			select {
			case <-ctx.Done():
				close(jobs)
				wg.Wait()
				return BatchJudgeResult{}, ctx.Err()
			case jobs <- batchJudgeJob{Case: c, Repeat: repeat}:
			}
		}
	}
	close(jobs)
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		if results[i].CaseID != results[j].CaseID {
			return results[i].CaseID < results[j].CaseID
		}
		if results[i].Target != results[j].Target {
			return results[i].Target < results[j].Target
		}
		return results[i].Repeat < results[j].Repeat
	})
	summary, byTag, byModel := summarizeBatchJudge(results)
	return BatchJudgeResult{
		Name:      spec.Name,
		Version:   spec.Version,
		StartedAt: startedAt,
		EndedAt:   time.Now().UTC(),
		Results:   results,
		Summary:   summary,
		ByTag:     byTag,
		ByModel:   byModel,
	}, nil
}

type batchJudgeJob struct {
	Case   JudgeCase
	Repeat int
}

func (e *Evaluator) evaluateBatchCase(ctx context.Context, spec BatchJudgeSpec, job batchJudgeJob) JudgeCaseResult {
	startedAt := time.Now()
	mode := normalizeMode(spec.Mode)
	caseRubric := firstNonEmpty(job.Case.Rubric, spec.Rubric)
	casePrompt := firstNonEmpty(job.Case.TargetPrompt, spec.TargetPrompt)
	target := targetWithModelFallback(spec.Target, spec.TargetModel)
	judge := targetWithModelFallback(spec.Judge, spec.JudgeModel)
	targetLabel := target.labelOrDefault()
	if strings.TrimSpace(targetLabel) == "" && strings.TrimSpace(job.Case.TargetOutput) != "" {
		targetLabel = "saved-output"
	}
	result := JudgeCaseResult{
		CaseID:         job.Case.ID,
		CaseName:       job.Case.Name,
		Tags:           append([]string(nil), job.Case.Tags...),
		Target:         targetLabel,
		TargetProvider: target.Provider,
		Judge:          judge.labelOrDefault(),
		JudgeProvider:  judge.Provider,
		Mode:           mode,
		Repeat:         repeatOrOne(job.Repeat),
		Metadata:       copyMap(job.Case.Metadata),
	}

	var judgeResult JudgeResult
	var err error
	if strings.TrimSpace(job.Case.TargetOutput) != "" {
		judgeResult, err = e.Judge(ctx, JudgeInput{
			Input:         job.Case.Input,
			TargetOutput:  job.Case.TargetOutput,
			GroundTruth:   job.Case.GroundTruth,
			Rubric:        caseRubric,
			Judge:         judge,
			JudgeModel:    spec.JudgeModel,
			Mode:          mode,
			PassThreshold: spec.PassThreshold,
			JudgeOptions:  spec.JudgeOptions,
		})
	} else {
		judgeResult, err = e.Evaluate(ctx, EvaluateInput{
			Input:         job.Case.Input,
			GroundTruth:   job.Case.GroundTruth,
			Rubric:        caseRubric,
			TargetPrompt:  casePrompt,
			Target:        target,
			Judge:         judge,
			TargetModel:   spec.TargetModel,
			JudgeModel:    spec.JudgeModel,
			Mode:          mode,
			PassThreshold: spec.PassThreshold,
			TargetOptions: spec.TargetOptions,
			JudgeOptions:  spec.JudgeOptions,
		})
	}
	result.DurationMS = time.Since(startedAt).Milliseconds()
	if err != nil {
		result.Error = err.Error()
		result.TargetOutput = firstNonEmpty(judgeResult.TargetOutput, job.Case.TargetOutput)
		return result
	}

	result.Score = judgeResult.Score
	result.Rationale = judgeResult.Rationale
	result.Passed = judgeResult.Passed
	result.PassThreshold = judgeResult.PassThreshold
	result.JSON = judgeResult.JSON
	result.RawJudgeOutput = judgeResult.RawJudgeOutput
	result.TargetOutput = judgeResult.TargetOutput
	result.Result = &judgeResult
	return result
}

func summarizeBatchJudge(results []JudgeCaseResult) (BatchJudgeSummary, map[string]BatchJudgeSummary, map[string]BatchJudgeSummary) {
	var summary BatchJudgeSummary
	byTag := make(map[string]BatchJudgeSummary)
	byModel := make(map[string]BatchJudgeSummary)
	scoreSumsByTag := make(map[string]float64)
	scoreSumsByModel := make(map[string]float64)
	var scoreSum float64
	for _, result := range results {
		addBatchJudgeResult(&summary, result, &scoreSum)
		modelKey := result.Target
		if strings.TrimSpace(modelKey) == "" {
			modelKey = "saved-output"
		}
		modelSummary := byModel[modelKey]
		modelScoreSum := scoreSumsByModel[modelKey]
		addBatchJudgeResult(&modelSummary, result, &modelScoreSum)
		byModel[modelKey] = modelSummary
		scoreSumsByModel[modelKey] = modelScoreSum

		for _, tag := range result.Tags {
			tag = strings.TrimSpace(tag)
			if tag == "" {
				continue
			}
			tagSummary := byTag[tag]
			tagScoreSum := scoreSumsByTag[tag]
			addBatchJudgeResult(&tagSummary, result, &tagScoreSum)
			byTag[tag] = tagSummary
			scoreSumsByTag[tag] = tagScoreSum
		}
	}
	finalizeBatchJudgeSummary(&summary, scoreSum)
	for key, group := range byModel {
		finalizeBatchJudgeSummary(&group, scoreSumsByModel[key])
		byModel[key] = group
	}
	for key, group := range byTag {
		finalizeBatchJudgeSummary(&group, scoreSumsByTag[key])
		byTag[key] = group
	}
	if len(byTag) == 0 {
		byTag = nil
	}
	if len(byModel) == 0 {
		byModel = nil
	}
	return summary, byTag, byModel
}

func addBatchJudgeResult(summary *BatchJudgeSummary, result JudgeCaseResult, scoreSum *float64) {
	summary.Total++
	summary.DurationMS += result.DurationMS
	if result.Error != "" {
		summary.Errors++
		return
	}
	summary.ScoreCount++
	*scoreSum += result.Score
	if result.Passed {
		summary.Passed++
	} else {
		summary.Failed++
	}
}

func finalizeBatchJudgeSummary(summary *BatchJudgeSummary, scoreSum float64) {
	if summary.ScoreCount > 0 {
		summary.MeanScore = scoreSum / float64(summary.ScoreCount)
	}
}

func isZeroTarget(target Target) bool {
	if target.Provider != "" || target.ModelID != "" {
		return false
	}
	if target.ModelConfig == nil {
		return true
	}
	return target.ModelConfig.Provider == "" && target.ModelConfig.ID == "" && target.ModelConfig.Name == ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
