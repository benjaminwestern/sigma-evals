// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// WriteVarianceSamplesJSONL writes normalized variance samples as newline-delimited JSON.
func WriteVarianceSamplesJSONL(w io.Writer, samples []VarianceSample) error {
	encoder := json.NewEncoder(w)
	for _, sample := range samples {
		if err := encoder.Encode(sample); err != nil {
			return err
		}
	}
	return nil
}

// ReadVarianceSamplesJSONL reads normalized variance samples from newline-delimited JSON.
func ReadVarianceSamplesJSONL(r io.Reader) ([]VarianceSample, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var samples []VarianceSample
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var sample VarianceSample
		if err := json.Unmarshal([]byte(line), &sample); err != nil {
			return nil, fmt.Errorf("decode variance sample line %d: %w", lineNumber, err)
		}
		samples = append(samples, sample)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return samples, nil
}

// WriteRunResultSamplesJSONL writes the suite-run samples used for variance comparison.
func WriteRunResultSamplesJSONL(w io.Writer, run RunResult) error {
	return WriteVarianceSamplesJSONL(w, VarianceSamplesFromRunResult(run))
}

// WriteBatchJudgeSamplesJSONL writes the batch-judge samples used for variance comparison.
func WriteBatchJudgeSamplesJSONL(w io.Writer, run BatchJudgeResult) error {
	return WriteVarianceSamplesJSONL(w, VarianceSamplesFromBatchJudgeResult(run))
}

// WriteJudgeAlignmentSamplesJSONL writes the judge-alignment samples used for variance comparison.
func WriteJudgeAlignmentSamplesJSONL(w io.Writer, run JudgeAlignmentRunResult) error {
	return WriteVarianceSamplesJSONL(w, VarianceSamplesFromJudgeAlignmentResult(run))
}
