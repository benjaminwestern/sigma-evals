// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"errors"
	"fmt"
	"strings"

	"github.com/wintermi/sigma"
)

var (
	// ErrInvalidOutput indicates a model or judge returned content that cannot be
	// scored as plain visible text.
	ErrInvalidOutput = errors.New("invalid model output")
)

// AssistantText extracts visible text from a Sigma assistant message. Thinking
// blocks are ignored; non-text visible blocks are rejected so eval scoring does
// not silently skip tool calls or images.
func AssistantText(message sigma.AssistantMessage) (string, error) {
	var b strings.Builder
	for _, block := range message.Content {
		switch block.Type {
		case sigma.ContentBlockText:
			b.WriteString(block.Text)
		case sigma.ContentBlockThinking:
			continue
		default:
			return "", fmt.Errorf("%w: assistant returned non-text content block %q", ErrInvalidOutput, block.Type)
		}
	}
	text := strings.TrimSpace(b.String())
	if text == "" {
		return "", fmt.Errorf("%w: assistant returned no visible text", ErrInvalidOutput)
	}
	return text, nil
}
