// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals_test

import (
	"strings"
	"testing"

	sigmaevals "github.com/benjaminwestern/sigma-evals"
	"github.com/wintermi/sigma"
)

func TestLoadSuiteRejectsSuitesWithoutAddressableCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		json string
	}{
		{name: "missing suite name", json: `{"cases":[{"id":"case-1"}]}`},
		{name: "no cases", json: `{"name":"empty","cases":[]}`},
		{name: "case without id", json: `{"name":"bad-case","cases":[{"input":"hello"}]}`},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if _, err := sigmaevals.LoadSuite(strings.NewReader(tt.json)); err == nil {
				t.Fatalf("LoadSuite(%s) succeeded, want validation error", tt.json)
			}
		})
	}
}

func TestDefaultRendererSupportsMultiTurnChatAndTools(t *testing.T) {
	t.Parallel()

	suite, err := sigmaevals.LoadSuiteFile("examples/tools/file-read-tool-call.json")
	if err != nil {
		t.Fatal(err)
	}
	request, err := (sigmaevals.DefaultRenderer{}).Render(t.Context(), sigmaevals.RenderInput{Suite: suite, Case: suite.Cases[0]})
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != "read_file" {
		t.Fatalf("tools = %+v, want read_file tool", request.Tools)
	}

	chatSuite, err := sigmaevals.LoadSuiteFile("examples/chat/multi-turn-context.json")
	if err != nil {
		t.Fatal(err)
	}
	request, err = (sigmaevals.DefaultRenderer{}).Render(t.Context(), sigmaevals.RenderInput{Suite: chatSuite, Case: chatSuite.Cases[0]})
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("messages = %d, want multi-turn chat messages", len(request.Messages))
	}

	imageSuite, err := sigmaevals.LoadSuiteFile("examples/chat/image-input-url.json")
	if err != nil {
		t.Fatal(err)
	}
	request, err = (sigmaevals.DefaultRenderer{}).Render(t.Context(), sigmaevals.RenderInput{Suite: imageSuite, Case: imageSuite.Cases[0]})
	if err != nil {
		t.Fatal(err)
	}
	if got := request.Messages[0].Content[1]; got.Type != sigma.ContentBlockImage || got.ImageSource != "url" || got.URL == "" {
		t.Fatalf("image block = %+v, want URL image content block", got)
	}
}
