// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"encoding/json"
	"math"
	"strings"
)

// TokenLogprob is one provider token logprob entry.
type TokenLogprob struct {
	Token       string         `json:"token"`
	Logprob     float64        `json:"logprob"`
	Bytes       []byte         `json:"bytes,omitempty"`
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

// TokenLogprobsFromMetadata extracts OpenAI-compatible logprobs from assistant metadata.
func TokenLogprobsFromMetadata(metadata map[string]any) ([]TokenLogprob, bool) {
	if len(metadata) == 0 {
		return nil, false
	}
	return DecodeTokenLogprobs(metadata["logprobs"])
}

// DecodeTokenLogprobs decodes either an OpenAI logprobs envelope or a direct token-logprob array.
func DecodeTokenLogprobs(value any) ([]TokenLogprob, bool) {
	if value == nil {
		return nil, false
	}
	if logprobs, ok := value.([]TokenLogprob); ok {
		return logprobs, len(logprobs) > 0
	}

	data, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var envelope struct {
		Content []TokenLogprob `json:"content"`
	}
	if err := json.Unmarshal(data, &envelope); err == nil && len(envelope.Content) > 0 {
		return envelope.Content, true
	}
	var direct []TokenLogprob
	if err := json.Unmarshal(data, &direct); err == nil && len(direct) > 0 {
		return direct, true
	}
	return nil, false
}

// GEvalScore computes the expected 1-5 score from score-token logprobs.
func GEvalScore(logprobs []TokenLogprob) (float64, bool) {
	for _, logprob := range logprobs {
		if len(logprob.TopLogprobs) > 0 {
			if score, ok := weightedScore(logprob.TopLogprobs); ok {
				return score, true
			}
		}
		if score, ok := scoreToken(logprob.Token); ok {
			return score, true
		}
	}
	return 0, false
}

// GEvalScoreForOutput computes G-Eval from logprobs attached to the visible score token.
func GEvalScoreForOutput(logprobs []TokenLogprob, output string) (float64, bool) {
	expected := strings.TrimSpace(output)
	fallback, ok := scoreToken(expected)
	if !ok {
		return 0, false
	}
	for _, logprob := range logprobs {
		if strings.TrimSpace(logprob.Token) != expected {
			continue
		}
		if len(logprob.TopLogprobs) > 0 {
			if score, ok := weightedScore(logprob.TopLogprobs); ok {
				return score, true
			}
		}
		return fallback, true
	}
	return 0, false
}

func weightedScore(logprobs []TokenLogprob) (float64, bool) {
	var totalProb float64
	var weightedSum float64
	for _, logprob := range logprobs {
		score, ok := scoreToken(logprob.Token)
		if !ok {
			continue
		}
		probability := math.Exp(logprob.Logprob)
		if probability <= 0 || math.IsInf(probability, 0) || math.IsNaN(probability) {
			continue
		}
		totalProb += probability
		weightedSum += probability * score
	}
	if totalProb == 0 {
		return 0, false
	}
	return weightedSum / totalProb, true
}

func scoreToken(token string) (float64, bool) {
	switch strings.TrimSpace(token) {
	case "1":
		return 1, true
	case "2":
		return 2, true
	case "3":
		return 3, true
	case "4":
		return 4, true
	case "5":
		return 5, true
	default:
		return 0, false
	}
}
