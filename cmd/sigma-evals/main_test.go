// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"testing"

	"github.com/wintermi/sigma"
)

func TestRegisterCommonProviders(t *testing.T) {
	t.Parallel()

	registry := sigma.DefaultRegistry()
	if err := registerCommonProviders(registry); err != nil {
		t.Fatal(err)
	}
	for _, provider := range []sigma.ProviderID{sigma.ProviderAnthropic, sigma.ProviderGoogle, sigma.ProviderGoogleVertex, sigma.ProviderMistral, sigma.ProviderAmazonBedrock, sigma.ProviderXAI} {
		if _, ok := registry.TextProvider(provider); !ok {
			t.Fatalf("text provider %q was not registered", provider)
		}
	}
	if _, ok := registry.ImageProvider(sigma.ProviderOpenRouter); !ok {
		t.Fatal("openrouter image provider was not registered")
	}
}

func TestParseCacheRetention(t *testing.T) {
	t.Parallel()

	retention, err := parseCacheRetention("persistent")
	if err != nil {
		t.Fatal(err)
	}
	if retention != sigma.CacheRetentionPersistent {
		t.Fatalf("retention = %q, want persistent", retention)
	}
	if _, err := parseCacheRetention("forever"); err == nil {
		t.Fatal("parseCacheRetention accepted unknown value")
	}
}

func TestRequestOptionsAcceptSessionAndCacheRetention(t *testing.T) {
	t.Parallel()

	options, err := requestOptions(" suite-run ", "long")
	if err != nil {
		t.Fatal(err)
	}
	var applied sigma.Options
	for _, option := range options {
		option(&applied)
	}
	if applied.SessionID != "suite-run" {
		t.Fatalf("session id = %q, want suite-run", applied.SessionID)
	}
	if applied.CacheRetention != sigma.CacheRetentionLong {
		t.Fatalf("cache retention = %q, want long", applied.CacheRetention)
	}
}
