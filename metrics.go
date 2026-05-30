// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"math"
	"sort"
)

// RegressionMetrics summarizes numeric judge agreement against labels.
type RegressionMetrics struct {
	Count                int     `json:"count"`
	MeanAbsoluteError    float64 `json:"meanAbsoluteError,omitempty"`
	MeanSquaredError     float64 `json:"meanSquaredError,omitempty"`
	RootMeanSquaredError float64 `json:"rootMeanSquaredError,omitempty"`
	Pearson              float64 `json:"pearson,omitempty"`
	Spearman             float64 `json:"spearman,omitempty"`
}

// ClassificationMetrics summarizes binary pass/fail judge agreement.
type ClassificationMetrics struct {
	Count            int     `json:"count"`
	Accuracy         float64 `json:"accuracy,omitempty"`
	Precision        float64 `json:"precision,omitempty"`
	Recall           float64 `json:"recall,omitempty"`
	F1               float64 `json:"f1,omitempty"`
	BalancedAccuracy float64 `json:"balancedAccuracy,omitempty"`
	CohenKappa       float64 `json:"cohenKappa,omitempty"`
	TruePositive     int     `json:"truePositive,omitempty"`
	TrueNegative     int     `json:"trueNegative,omitempty"`
	FalsePositive    int     `json:"falsePositive,omitempty"`
	FalseNegative    int     `json:"falseNegative,omitempty"`
}

// CalibrationMetrics summarizes score-as-probability judge calibration.
type CalibrationMetrics struct {
	Count      int     `json:"count"`
	BrierScore float64 `json:"brierScore,omitempty"`
}

// ComputeRegressionMetrics returns MAE, MSE, RMSE, Pearson, and Spearman for paired scores.
func ComputeRegressionMetrics(expected []float64, actual []float64) RegressionMetrics {
	count := min(len(expected), len(actual))
	metrics := RegressionMetrics{Count: count}
	if count == 0 {
		return metrics
	}

	var absSum, sqSum float64
	for i := 0; i < count; i++ {
		diff := actual[i] - expected[i]
		absSum += math.Abs(diff)
		sqSum += diff * diff
	}
	metrics.MeanAbsoluteError = absSum / float64(count)
	metrics.MeanSquaredError = sqSum / float64(count)
	metrics.RootMeanSquaredError = math.Sqrt(metrics.MeanSquaredError)
	metrics.Pearson = pearson(expected[:count], actual[:count])
	metrics.Spearman = pearson(ranks(expected[:count]), ranks(actual[:count]))
	return metrics
}

// ComputeClassificationMetrics returns confusion-matrix metrics and Cohen's kappa.
func ComputeClassificationMetrics(expected []bool, actual []bool) ClassificationMetrics {
	count := min(len(expected), len(actual))
	metrics := ClassificationMetrics{Count: count}
	if count == 0 {
		return metrics
	}

	for i := 0; i < count; i++ {
		switch {
		case expected[i] && actual[i]:
			metrics.TruePositive++
		case !expected[i] && !actual[i]:
			metrics.TrueNegative++
		case !expected[i] && actual[i]:
			metrics.FalsePositive++
		case expected[i] && !actual[i]:
			metrics.FalseNegative++
		}
	}

	correct := metrics.TruePositive + metrics.TrueNegative
	metrics.Accuracy = float64(correct) / float64(count)
	metrics.Precision = safeDiv(float64(metrics.TruePositive), float64(metrics.TruePositive+metrics.FalsePositive))
	metrics.Recall = safeDiv(float64(metrics.TruePositive), float64(metrics.TruePositive+metrics.FalseNegative))
	metrics.F1 = safeDiv(2*metrics.Precision*metrics.Recall, metrics.Precision+metrics.Recall)
	specificity := safeDiv(float64(metrics.TrueNegative), float64(metrics.TrueNegative+metrics.FalsePositive))
	metrics.BalancedAccuracy = (metrics.Recall + specificity) / 2

	expectedPositive := metrics.TruePositive + metrics.FalseNegative
	expectedNegative := metrics.TrueNegative + metrics.FalsePositive
	actualPositive := metrics.TruePositive + metrics.FalsePositive
	actualNegative := metrics.TrueNegative + metrics.FalseNegative
	chanceAgreement := safeDiv(float64(expectedPositive*actualPositive+expectedNegative*actualNegative), float64(count*count))
	metrics.CohenKappa = safeDiv(metrics.Accuracy-chanceAgreement, 1-chanceAgreement)
	return metrics
}

// ComputeCalibrationMetrics computes Brier score after mapping scores onto [0, 1].
func ComputeCalibrationMetrics(expectedPassed []bool, actualScores []float64, minScore float64, maxScore float64) CalibrationMetrics {
	count := min(len(expectedPassed), len(actualScores))
	metrics := CalibrationMetrics{Count: count}
	if count == 0 || maxScore <= minScore {
		return metrics
	}
	var sum float64
	for i := 0; i < count; i++ {
		probability := (actualScores[i] - minScore) / (maxScore - minScore)
		probability = math.Max(0, math.Min(1, probability))
		expected := 0.0
		if expectedPassed[i] {
			expected = 1
		}
		diff := probability - expected
		sum += diff * diff
	}
	metrics.BrierScore = sum / float64(count)
	return metrics
}

func pearson(x []float64, y []float64) float64 {
	count := min(len(x), len(y))
	if count < 2 {
		return 0
	}
	var sumX, sumY float64
	for i := 0; i < count; i++ {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / float64(count)
	meanY := sumY / float64(count)

	var numerator, denomX, denomY float64
	for i := 0; i < count; i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		denomX += dx * dx
		denomY += dy * dy
	}
	return safeDiv(numerator, math.Sqrt(denomX*denomY))
}

func ranks(values []float64) []float64 {
	type item struct {
		index int
		value float64
	}
	items := make([]item, len(values))
	for i, value := range values {
		items[i] = item{index: i, value: value}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].value < items[j].value })

	out := make([]float64, len(values))
	for i := 0; i < len(items); {
		j := i + 1
		for j < len(items) && items[j].value == items[i].value {
			j++
		}
		// Average rank for ties. Ranks are 1-based.
		rank := float64(i+1+j) / 2
		for k := i; k < j; k++ {
			out[items[k].index] = rank
		}
		i = j
	}
	return out
}

func safeDiv(numerator float64, denominator float64) float64 {
	if denominator == 0 || math.IsNaN(denominator) {
		return 0
	}
	return numerator / denominator
}
