// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"context"
	"fmt"
	"strings"

	"github.com/wintermi/sigma"
)

// RenderInput gives a renderer access to both suite-level defaults and the case.
type RenderInput struct {
	Suite Suite `json:"suite"`
	Case  Case  `json:"case"`
}

// Renderer turns a case into the exact Sigma request sent to each model.
type Renderer interface {
	Render(context.Context, RenderInput) (sigma.Request, error)
}

// RendererFunc adapts a function into a Renderer.
type RendererFunc func(context.Context, RenderInput) (sigma.Request, error)

// Render calls f(ctx, input).
func (f RendererFunc) Render(ctx context.Context, input RenderInput) (sigma.Request, error) {
	return f(ctx, input)
}

// DefaultRenderer renders Case.Messages when present, otherwise Case.Input as a
// single user text message. Case.SystemPrompt overrides Suite.SystemPrompt.
type DefaultRenderer struct{}

// Render implements Renderer.
func (DefaultRenderer) Render(_ context.Context, input RenderInput) (sigma.Request, error) {
	if strings.TrimSpace(input.Case.ID) == "" {
		return sigma.Request{}, fmt.Errorf("case id is required")
	}

	systemPrompt := strings.TrimSpace(input.Case.SystemPrompt)
	if systemPrompt == "" {
		systemPrompt = strings.TrimSpace(input.Suite.SystemPrompt)
	}

	messages := append([]sigma.Message(nil), input.Case.Messages...)
	if len(messages) == 0 {
		messages = []sigma.Message{sigma.UserText(input.Case.Input)}
	}

	return sigma.Request{
		SystemPrompt: systemPrompt,
		Messages:     messages,
		Tools:        append([]sigma.Tool(nil), input.Case.Tools...),
	}, nil
}

func rendererOrDefault(renderer Renderer) Renderer {
	if renderer == nil {
		return DefaultRenderer{}
	}
	return renderer
}
