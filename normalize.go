// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"strings"
	"unicode"
)

// NormalizeAnswer applies a small TriviaQA-style normalizer suitable for alias
// matching: lowercase, remove punctuation and English articles, and collapse
// whitespace.
func NormalizeAnswer(text string) string {
	text = strings.ToLower(strings.ReplaceAll(text, "_", " "))
	var b strings.Builder
	for _, r := range text {
		switch {
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			b.WriteRune(' ')
		default:
			b.WriteRune(r)
		}
	}

	parts := strings.Fields(b.String())
	kept := parts[:0]
	for _, part := range parts {
		if part == "a" || part == "an" || part == "the" {
			continue
		}
		kept = append(kept, part)
	}
	return strings.Join(kept, " ")
}

func tokenF1(prediction string, groundTruth string) float64 {
	predTokens := strings.Fields(NormalizeAnswer(prediction))
	truthTokens := strings.Fields(NormalizeAnswer(groundTruth))
	if len(predTokens) == 0 || len(truthTokens) == 0 {
		if len(predTokens) == len(truthTokens) {
			return 1
		}
		return 0
	}

	truthCounts := make(map[string]int, len(truthTokens))
	for _, token := range truthTokens {
		truthCounts[token]++
	}
	common := 0
	for _, token := range predTokens {
		if truthCounts[token] > 0 {
			common++
			truthCounts[token]--
		}
	}
	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(predTokens))
	recall := float64(common) / float64(len(truthTokens))
	return 2 * precision * recall / (precision + recall)
}
