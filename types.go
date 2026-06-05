// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"time"

	"github.com/wintermi/sigma"
)

// Completer is the subset of sigma.Client used by the harness.
type Completer interface {
	Complete(context.Context, sigma.Model, sigma.Request, ...sigma.Option) (sigma.AssistantMessage, error)
}

// EvalDataType identifies the part of a task run being evaluated.
type EvalDataType string

const (
	// EvalDataFinalAnswer evaluates only the final assistant answer.
	EvalDataFinalAnswer EvalDataType = "final_answer"
	// EvalDataFullTrace evaluates the conversation/tool trace that produced the answer.
	EvalDataFullTrace EvalDataType = "full_trace"
	// EvalDataReferenceAnswer evaluates an output against a reference answer.
	EvalDataReferenceAnswer EvalDataType = "reference_answer"
)

// Suite groups related cases that should be run with the same rendering and
// scoring rules.
type Suite struct {
	Name         string         `json:"name"`
	Version      string         `json:"version,omitempty"`
	Description  string         `json:"description,omitempty"`
	SystemPrompt string         `json:"systemPrompt,omitempty"`
	DataType     EvalDataType   `json:"dataType,omitempty"`
	Cases        []Case         `json:"cases"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// Case is one provider-neutral evaluation input.
type Case struct {
	ID           string          `json:"id"`
	Name         string          `json:"name,omitempty"`
	SystemPrompt string          `json:"systemPrompt,omitempty"`
	DataType     EvalDataType    `json:"dataType,omitempty"`
	Input        string          `json:"input,omitempty"`
	Messages     []sigma.Message `json:"messages,omitempty"`
	Tools        []sigma.Tool    `json:"tools,omitempty"`
	Trace        Trace           `json:"trace,omitempty"`
	Expected     Expected        `json:"expected,omitempty"`
	Tags         []string        `json:"tags,omitempty"`
	Metadata     map[string]any  `json:"metadata,omitempty"`
}

// Trace records existing conversation/tool activity for full-trace evals.
type Trace struct {
	Messages []sigma.Message `json:"messages,omitempty"`
	Events   []TraceEvent    `json:"events,omitempty"`
	Metadata map[string]any  `json:"metadata,omitempty"`
}

// TraceEvent is an opaque named event in a task run trace.
type TraceEvent struct {
	Type     string         `json:"type"`
	Name     string         `json:"name,omitempty"`
	Payload  any            `json:"payload,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Expected describes the target output contract for deterministic scorers and
// judge prompts.
type Expected struct {
	Output          string             `json:"output,omitempty"`
	Answers         []string           `json:"answers,omitempty"`
	NegativeAnswers []string           `json:"negativeAnswers,omitempty"`
	Patterns        []string           `json:"patterns,omitempty"`
	JSON            any                `json:"json,omitempty"`
	Choices         []Choice           `json:"choices,omitempty"`
	CorrectChoices  []string           `json:"correctChoices,omitempty"`
	ToolCalls       []ExpectedToolCall `json:"toolCalls,omitempty"`
	Rubric          string             `json:"rubric,omitempty"`
	Metadata        map[string]any     `json:"metadata,omitempty"`
}

// Choice describes one multiple-choice option.
type Choice struct {
	Label string `json:"label"`
	Text  string `json:"text"`
}

// ExpectedToolCall describes an expected assistant tool call. Arguments are
// compared structurally when set, and ignored when nil.
type ExpectedToolCall struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

// RunSpec configures one suite run across one or more models.
type RunSpec struct {
	Suite       Suite          `json:"suite"`
	Models      []sigma.Model  `json:"models"`
	Renderer    Renderer       `json:"-"`
	Scorers     []Scorer       `json:"-"`
	Options     []sigma.Option `json:"-"`
	Repeats     int            `json:"repeats,omitempty"`
	Concurrency int            `json:"concurrency,omitempty"`
}

// RunResult is the portable output of a suite run.
type RunResult struct {
	SuiteName    string                  `json:"suiteName"`
	SuiteVersion string                  `json:"suiteVersion,omitempty"`
	StartedAt    time.Time               `json:"startedAt"`
	EndedAt      time.Time               `json:"endedAt"`
	Results      []CaseResult            `json:"results"`
	Summary      Summary                 `json:"summary"`
	Metadata     map[string]any          `json:"metadata,omitempty"`
	ByModel      map[string]ModelSummary `json:"byModel,omitempty"`
	ByTag        map[string]Summary      `json:"byTag,omitempty"`
}

// CaseResult records one target model attempt and all scores attached to it.
type CaseResult struct {
	CaseID        string                 `json:"caseId"`
	CaseName      string                 `json:"caseName,omitempty"`
	Tags          []string               `json:"tags,omitempty"`
	Model         string                 `json:"model"`
	Provider      sigma.ProviderID       `json:"provider,omitempty"`
	Repeat        int                    `json:"repeat"`
	Request       sigma.Request          `json:"request,omitempty"`
	Output        string                 `json:"output,omitempty"`
	Message       sigma.AssistantMessage `json:"message,omitempty"`
	Scores        []Score                `json:"scores,omitempty"`
	Error         string                 `json:"error,omitempty"`
	ErrorDetails  *ErrorDetails          `json:"errorDetails,omitempty"`
	DurationMS    int64                  `json:"durationMs"`
	Usage         *sigma.Usage           `json:"usage,omitempty"`
	Cost          *sigma.Cost            `json:"cost,omitempty"`
	ProviderMeta  map[string]any         `json:"providerMetadata,omitempty"`
	ScorerVersion string                 `json:"scorerVersion,omitempty"`
}

// Score is a normalized scorer or judge result.
type Score struct {
	Name      string         `json:"name"`
	Score     float64        `json:"score"`
	Passed    bool           `json:"passed"`
	Rationale string         `json:"rationale,omitempty"`
	Details   map[string]any `json:"details,omitempty"`
}

// Summary aggregates a run.
type Summary struct {
	Total      int `json:"total"`
	Passed     int `json:"passed"`
	Failed     int `json:"failed"`
	Errors     int `json:"errors"`
	ScoreCount int `json:"scoreCount"`
}

// ModelSummary aggregates a run for one model.
type ModelSummary struct {
	Total      int     `json:"total"`
	Passed     int     `json:"passed"`
	Failed     int     `json:"failed"`
	Errors     int     `json:"errors"`
	MeanScore  float64 `json:"meanScore,omitempty"`
	ScoreCount int     `json:"scoreCount"`
	DurationMS int64   `json:"durationMs"`
}
