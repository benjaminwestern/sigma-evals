// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"context"
	"math"
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestMultipleChoiceScorerExtractsLabelsAndAnswerText(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		output string
		passed bool
	}{
		{name: "label", output: "The answer is A.", passed: true},
		{name: "answer text", output: "I would choose Paris.", passed: true},
		{name: "wrong label", output: "The answer is B.", passed: false},
		{name: "no choice", output: "I am not sure.", passed: false},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			score, err := sigmaevals.MultipleChoiceScorer{}.Score(context.Background(), sigmaevals.ScoreInput{
				Case: sigmaevals.Case{ID: "mc", Expected: sigmaevals.Expected{
					Choices:        []sigmaevals.Choice{{Label: "A", Text: "Paris"}, {Label: "B", Text: "London"}},
					CorrectChoices: []string{"A"},
				}},
				Output: tt.output,
			})
			if err != nil {
				t.Fatal(err)
			}
			if score.Passed != tt.passed {
				t.Fatalf("score = %+v, want passed=%v", score, tt.passed)
			}
		})
	}
}

func TestPassAtKEstimator(t *testing.T) {
	t.Parallel()

	got := sigmaevals.EstimatePassAtK(10, 2, 1)
	if math.Abs(got-0.2) > 0.0001 {
		t.Fatalf("EstimatePassAtK = %.4f, want 0.2", got)
	}
	if sigmaevals.PassAtK([]bool{false, true, false}, 3) != 1 {
		t.Fatal("PassAtK should be 1 when n-c < k")
	}
}

func TestPairwiseJudgeChecksSwappedOrderConsistency(t *testing.T) {
	t.Parallel()

	client := &scriptedClient{responses: []sigma.AssistantMessage{
		textMessage(`{"winner":"A","rationale":"A is exact"}`),
		textMessage(`{"winner":"B","rationale":"B is exact after swap"}`),
	}}
	result, err := sigmaevals.NewEvaluator(client).PairwiseJudge(context.Background(), sigmaevals.PairwiseJudgeInput{
		Input:      "Translate hello to French.",
		AnswerA:    "Bonjour",
		AnswerB:    "Hola",
		Reference:  "Bonjour",
		Rubric:     "Pick the answer that matches the reference.",
		JudgeModel: sigma.Model{ID: "judge", Provider: "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Consistent || result.Winner != sigmaevals.PairwiseWinnerA {
		t.Fatalf("result = %+v, want consistent A winner", result)
	}
}

func TestRubricGEvalScoreForOutputScoresEachMetricInRange(t *testing.T) {
	t.Parallel()

	rubric := sigmaevals.Rubric{Dimensions: []sigmaevals.RubricDimension{
		{Name: "Accuracy", Type: sigmaevals.RatingFiveStar, Weight: 3},
		{Name: "Safety", Type: sigmaevals.RatingPassFail, Weight: 1},
	}}
	logprobs := []sigmaevals.TokenLogprob{
		{Token: "{\"accuracy\":"},
		{Token: "4", Logprob: math.Log(0.8), TopLogprobs: []sigmaevals.TokenLogprob{{Token: "4", Logprob: math.Log(0.8)}, {Token: "5", Logprob: math.Log(0.2)}}},
		{Token: ",\"safety\":\""},
		{Token: "pass", Logprob: math.Log(0.9), TopLogprobs: []sigmaevals.TokenLogprob{{Token: "pass", Logprob: math.Log(0.9)}, {Token: "fail", Logprob: math.Log(0.1)}}},
		{Token: "\"}"},
	}
	scores, err := sigmaevals.RubricGEvalScoreForOutput(rubric, "", logprobs)
	if err != nil {
		t.Fatal(err)
	}
	if scores.Scores["accuracy"] <= 4 || scores.Scores["accuracy"] >= 5 {
		t.Fatalf("accuracy = %.4f, want weighted score between 4 and 5", scores.Scores["accuracy"])
	}
	if scores.Normalized["safety"] <= 0.8 {
		t.Fatalf("safety = %.4f, want high pass probability", scores.Normalized["safety"])
	}
}

func TestPresetRubricCarriesDetailsIntoPromptAndRejectsUnknown(t *testing.T) {
	t.Parallel()

	rubric, err := sigmaevals.PresetRubric(sigmaevals.PresetToolCall, "Require lookup_fact before final answer.")
	if err != nil {
		t.Fatal(err)
	}
	prompt := rubric.Prompt(sigmaevals.ScoreInput{
		Case:   sigmaevals.Case{ID: "tool", Input: "Find the deployment ticket.", Expected: sigmaevals.Expected{Rubric: "Do not accept guessed values."}},
		Output: "TICKET-7429",
	})
	if !strings.Contains(prompt, "Require lookup_fact before final answer.") || !strings.Contains(prompt, "Do not accept guessed values.") {
		t.Fatalf("prompt = %q, want preset details and case rubric", prompt)
	}

	if _, err := sigmaevals.PresetRubric("unknown", ""); err == nil {
		t.Fatal("PresetRubric accepted an unknown preset")
	}
}

func TestScoreExistingScoresWithoutTargetModelCall(t *testing.T) {
	t.Parallel()

	result, err := sigmaevals.ScoreExisting(context.Background(), sigmaevals.ScoreExistingSpec{
		Suite: sigmaevals.Suite{Name: "existing", Cases: []sigmaevals.Case{
			{ID: "c1", Tags: []string{"tag-a"}, Expected: sigmaevals.Expected{Answers: []string{"Bonjour"}}},
		}},
		Outputs: []sigmaevals.ExistingOutput{{CaseID: "c1", Model: sigma.Model{ID: "m", Provider: "test"}, Output: "Bonjour"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.Passed != 1 || result.ByTag["tag-a"].Passed != 1 {
		t.Fatalf("result = %+v, want scored existing output and tag summary", result)
	}
}
