// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import "github.com/wintermi/sigma"

func withStructuredOutput(responseFormat any) sigma.Option {
	return withMergedOpenAIOptions(func(options *sigma.OpenAIOptions) {
		options.ResponseFormat = responseFormat
	})
}

func withOpenAILogprobs(topLogprobs int) sigma.Option {
	if topLogprobs <= 0 {
		return func(*sigma.Options) {}
	}
	return withMergedOpenAIOptions(func(options *sigma.OpenAIOptions) {
		options.TopLogprobs = topLogprobs
	})
}

func withMergedOpenAIOptions(update func(*sigma.OpenAIOptions)) sigma.Option {
	return func(options *sigma.Options) {
		openAIOptions := sigma.OpenAIOptions{}
		if options.OpenAIOptions != nil {
			openAIOptions = *options.OpenAIOptions
			if options.OpenAIOptions.ParallelToolCalls != nil {
				parallelToolCalls := *options.OpenAIOptions.ParallelToolCalls
				openAIOptions.ParallelToolCalls = &parallelToolCalls
			}
		}
		update(&openAIOptions)
		options.OpenAIOptions = &openAIOptions
	}
}

func copyMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	copied := make(map[string]any, len(values))
	for key, value := range values {
		copied[key] = value
	}
	return copied
}
