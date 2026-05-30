// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

// EstimatePassAtK estimates pass@k from total samples n, correct samples c, and k.
// It uses the standard unbiased estimator: 1 - C(n-c, k) / C(n, k).
func EstimatePassAtK(total int, correct int, k int) float64 {
	if total <= 0 || k <= 0 || correct <= 0 {
		return 0
	}
	if correct > total {
		correct = total
	}
	if k > total {
		k = total
	}
	if total-correct < k {
		return 1
	}
	probNoCorrect := 1.0
	for i := total - correct + 1; i <= total; i++ {
		probNoCorrect *= 1 - float64(k)/float64(i)
	}
	return 1 - probNoCorrect
}

// PassAtK estimates pass@k from individual sample correctness flags.
func PassAtK(correct []bool, k int) float64 {
	count := 0
	for _, ok := range correct {
		if ok {
			count++
		}
	}
	return EstimatePassAtK(len(correct), count, k)
}
