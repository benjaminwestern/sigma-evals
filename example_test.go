// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"fmt"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func ExampleRunner_Run() {
	suite, err := sigmaevals.LoadSuiteFile("examples/generic/answer-aliases.json")
	if err != nil {
		panic(err)
	}

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage("Northstar"),
		textMessage("#ops-critical"),
		textMessage("Cedar"),
	}}
	model := sigma.Model{ID: "example-model", Provider: "example", Name: "example-model"}

	result, err := sigmaevals.NewRunner(client).Run(context.Background(), sigmaevals.RunSpec{
		Suite:       suite,
		Models:      []sigma.Model{model},
		Scorers:     []sigmaevals.Scorer{sigmaevals.AnswerScorer{Mode: sigmaevals.MatchContains}},
		Concurrency: 1,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s: %d/%d passed\n", result.SuiteName, result.Summary.Passed, result.Summary.Total)
	// Output: Answer Aliases: 3/3 passed
}

func ExampleEvaluator_EvaluateJudges() {
	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage(`{"score":1,"rationale":"exact answer","passed":true}`),
		textMessage(`{"score":0,"rationale":"wrong ticket","passed":false}`),
	}}
	judge := sigma.Model{ID: "example-judge", Provider: "example", Name: "example-judge"}

	result, err := sigmaevals.NewEvaluator(client).EvaluateJudges(context.Background(), sigmaevals.JudgeAlignmentSpec{
		Name: "judge-alignment-smoke",
		Cases: []sigmaevals.JudgeAlignmentCase{
			{
				ID:             "correct-ticket",
				Input:          "The deployment ticket is TICKET-7429.",
				TargetOutput:   "TICKET-7429",
				GroundTruth:    "TICKET-7429",
				Rubric:         "Return score 1 only for the exact ticket value.",
				ExpectedScore:  1,
				ExpectedPassed: true,
			},
			{
				ID:             "wrong-ticket",
				Input:          "The deployment ticket is TICKET-7429.",
				TargetOutput:   "TICKET-1001",
				GroundTruth:    "TICKET-7429",
				Rubric:         "Return score 1 only for the exact ticket value.",
				ExpectedScore:  0,
				ExpectedPassed: false,
			},
		},
		JudgeModels: []sigma.Model{judge},
		Tolerance:   0.01,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("judge classification accuracy: %.1f\n", result.Summary.Classification.Accuracy)
	// Output: judge classification accuracy: 1.0
}
