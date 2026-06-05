// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"strings"
	"time"

	"github.com/wintermi/sigma"
)

// ErrorDetails records provider-neutral error classification data for a failed
// model attempt. It mirrors Sigma's typed classifier while keeping result JSON
// portable and free of the raw Go error value.
type ErrorDetails struct {
	Class        sigma.ErrorClass `json:"class,omitempty"`
	Provider     sigma.ProviderID `json:"provider,omitempty"`
	API          sigma.API        `json:"api,omitempty"`
	Model        sigma.ModelID    `json:"model,omitempty"`
	StatusCode   int              `json:"statusCode,omitempty"`
	ProviderCode string           `json:"providerCode,omitempty"`
	Message      string           `json:"message,omitempty"`
	RequestID    string           `json:"requestId,omitempty"`
	Retryable    bool             `json:"retryable,omitempty"`
	RetryAfterMS int64            `json:"retryAfterMs,omitempty"`
}

func classifyError(err error) *ErrorDetails {
	if err == nil {
		return nil
	}
	classification := sigma.ClassifyError(err)
	details := ErrorDetails{
		Class:        classification.Class,
		Provider:     classification.Provider,
		API:          classification.API,
		Model:        classification.Model,
		StatusCode:   classification.StatusCode,
		ProviderCode: classification.ProviderCode,
		Message:      classification.Message,
		RequestID:    classification.RequestID,
		Retryable:    classification.RetryHint.Retryable,
		RetryAfterMS: retryAfterMilliseconds(classification.RetryHint.After),
	}
	if strings.TrimSpace(details.Message) == "" {
		details.Message = err.Error()
	}
	return &details
}

func errorDetailsFromMessage(message string) *ErrorDetails {
	message = strings.TrimSpace(message)
	if message == "" {
		return nil
	}
	return &ErrorDetails{Class: sigma.ErrorClassUnknown, Message: message}
}

func classifyErrorOrMessage(err error, message string) *ErrorDetails {
	if details := classifyError(err); details != nil {
		return details
	}
	return errorDetailsFromMessage(message)
}

func retryAfterMilliseconds(after time.Duration) int64 {
	if after <= 0 {
		return 0
	}
	return after.Milliseconds()
}
