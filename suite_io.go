// Copyright (c) 2026 Benjamin Western
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigmaevals

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// LoadSuite decodes a Suite from JSON.
func LoadSuite(reader io.Reader) (Suite, error) {
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	var suite Suite
	if err := decoder.Decode(&suite); err != nil {
		return Suite{}, err
	}
	if strings.TrimSpace(suite.Name) == "" {
		return Suite{}, fmt.Errorf("suite name is required")
	}
	if len(suite.Cases) == 0 {
		return Suite{}, fmt.Errorf("suite %q must contain at least one case", suite.Name)
	}
	for i, c := range suite.Cases {
		if strings.TrimSpace(c.ID) == "" {
			return Suite{}, fmt.Errorf("suite %q case %d has no id", suite.Name, i)
		}
	}
	return suite, nil
}

// LoadSuiteFile decodes a Suite from a JSON file.
func LoadSuiteFile(path string) (Suite, error) {
	file, err := os.Open(path)
	if err != nil {
		return Suite{}, err
	}
	defer file.Close()
	return LoadSuite(file)
}
