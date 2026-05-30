// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import "github.com/wintermi/sigma"

const opencodeAPIMetadataKey = "opencodeAPI"

func withStructuredOutput(model sigma.Model, responseFormat map[string]any) sigma.Option {
	if actualAPI(model) == sigma.APIOpenAIResponses {
		return withOpenAIProviderOptions(map[string]any{
			"text": map[string]any{"format": responsesTextFormat(responseFormat)},
		})
	}
	return withOpenAIExtraBody(map[string]any{"response_format": responseFormat})
}

func withOpenAILogprobs(topLogprobs int) sigma.Option {
	return withOpenAIExtraBody(map[string]any{
		"logprobs":     true,
		"top_logprobs": topLogprobs,
	})
}

func withOpenAIExtraBody(values map[string]any) sigma.Option {
	return func(options *sigma.Options) {
		if len(values) == 0 {
			return
		}
		providerOptions := ensureOpenAIProviderOptions(options)
		extraBody, _ := providerOptions["extra_body"].(map[string]any)
		if extraBody == nil {
			extraBody = make(map[string]any, len(values))
			providerOptions["extra_body"] = extraBody
		}
		for key, value := range values {
			extraBody[key] = value
		}
	}
}

func withOpenAIProviderOptions(values map[string]any) sigma.Option {
	return func(options *sigma.Options) {
		if len(values) == 0 {
			return
		}
		providerOptions := ensureOpenAIProviderOptions(options)
		for key, value := range values {
			providerOptions[key] = value
		}
	}
}

func ensureOpenAIProviderOptions(options *sigma.Options) map[string]any {
	if options.ProviderOptions == nil {
		options.ProviderOptions = make(map[sigma.ProviderID]map[string]any)
	}
	providerOptions := options.ProviderOptions[sigma.ProviderOpenAI]
	if providerOptions == nil {
		providerOptions = make(map[string]any)
		options.ProviderOptions[sigma.ProviderOpenAI] = providerOptions
	}
	return providerOptions
}

func responsesTextFormat(responseFormat map[string]any) map[string]any {
	if responseFormat == nil {
		return nil
	}
	formatType, _ := responseFormat["type"].(string)
	if formatType != "json_schema" {
		return copyMap(responseFormat)
	}
	jsonSchema, _ := responseFormat["json_schema"].(map[string]any)
	if jsonSchema == nil {
		return copyMap(responseFormat)
	}
	textFormat := map[string]any{"type": "json_schema"}
	for key, value := range jsonSchema {
		textFormat[key] = value
	}
	return textFormat
}

func actualAPI(model sigma.Model) sigma.API {
	if model.ProviderMetadata != nil {
		if api, ok := model.ProviderMetadata[opencodeAPIMetadataKey].(string); ok && api != "" {
			return sigma.API(api)
		}
	}
	return model.API
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
