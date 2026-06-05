// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
)

func TestVarianceSamplesJSONLRoundTrip(t *testing.T) {
	t.Parallel()

	var b strings.Builder
	input := []sigmaevals.VarianceSample{
		{Layer: sigmaevals.VarianceLayerSuite, CaseID: "case-1", Model: "model-a", Scorer: "exact", Score: 1, Passed: true},
		{Layer: sigmaevals.VarianceLayerBatchJudge, CaseID: "case-2", Model: "model-b", Scorer: "g_eval", Score: 4.5, Passed: true},
	}
	if err := sigmaevals.WriteVarianceSamplesJSONL(&b, input); err != nil {
		t.Fatal(err)
	}
	output, err := sigmaevals.ReadVarianceSamplesJSONL(strings.NewReader(b.String()))
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 2 || output[1].Layer != sigmaevals.VarianceLayerBatchJudge || output[1].Score != 4.5 {
		t.Fatalf("output = %+v, want round-tripped samples", output)
	}
}

func TestReadVarianceSamplesJSONLReportsLineNumber(t *testing.T) {
	t.Parallel()

	_, err := sigmaevals.ReadVarianceSamplesJSONL(strings.NewReader("{}\nnot-json\n"))
	if err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("err = %v, want line number", err)
	}
}
