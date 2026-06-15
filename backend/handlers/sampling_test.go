package handlers

import (
	"testing"

	"github.com/navy1999/spectra-rag/backend/llmcaps"
)

func TestSamplingPayload_GatesUnsupported(t *testing.T) {
	caps := llmcaps.NewFromMap(map[string][]string{
		"good":    {"temperature", "top_p", "frequency_penalty", "presence_penalty"},
		"minimal": {"temperature"},
	})
	p := SamplingProfile{Temperature: 0.3, TopP: f64(0.9), PresencePenalty: f64(0.2), FrequencyPenalty: f64(0.5)}

	full := samplingPayload("good", p, caps)
	for _, k := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty"} {
		if _, ok := full[k]; !ok {
			t.Errorf("model 'good' should send %s", k)
		}
	}

	min := samplingPayload("minimal", p, caps)
	if _, ok := min["top_p"]; ok {
		t.Error("model 'minimal' must not get top_p")
	}
	if _, ok := min["frequency_penalty"]; ok {
		t.Error("model 'minimal' must not get frequency_penalty")
	}
	if v, ok := min["temperature"]; !ok || v != 0.3 {
		t.Error("temperature is always sent")
	}
}

func TestSamplingPayload_NilCapsTemperatureOnly(t *testing.T) {
	p := SamplingProfile{Temperature: 0.3, TopP: f64(0.9), FrequencyPenalty: f64(0.5)}
	out := samplingPayload("anything", p, nil)
	if len(out) != 1 {
		t.Errorf("nil caps must send only temperature, got %v", out)
	}
}
