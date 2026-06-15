package llmcaps

import "testing"

const sampleModels = `{"data":[
{"id":"liquid/lfm-2.5-1.2b-instruct:free","supported_parameters":["temperature","top_p","top_k","min_p","frequency_penalty","presence_penalty","repetition_penalty","seed"]},
{"id":"nex-agi/nex-n2-pro:free","supported_parameters":["temperature","top_p","frequency_penalty","logprobs"]}
]}`

func TestParseAndSupports(t *testing.T) {
	c, err := parse([]byte(sampleModels))
	if err != nil {
		t.Fatal(err)
	}
	lfm := "liquid/lfm-2.5-1.2b-instruct:free"
	if !c.Supports(lfm, "frequency_penalty") {
		t.Error("LFM should support frequency_penalty")
	}
	if c.Supports(lfm, "logit_bias") {
		t.Error("LFM should NOT support logit_bias")
	}
	if c.Supports(lfm, "logprobs") {
		t.Error("LFM should NOT support logprobs")
	}
	if !c.Supports("nex-agi/nex-n2-pro:free", "logprobs") {
		t.Error("nex should support logprobs")
	}
	if c.Supports("nex-agi/nex-n2-pro:free", "presence_penalty") {
		t.Error("nex should NOT support presence_penalty")
	}
	if !c.Known(lfm) || c.Known("unknown/model") {
		t.Error("Known() wrong")
	}
	if c.Supports("unknown/model", "temperature") {
		t.Error("unknown model must support nothing")
	}
}

func TestParseEmpty(t *testing.T) {
	if _, err := parse([]byte(`{"data":[]}`)); err == nil {
		t.Error("empty model list should error")
	}
}

func TestNilSafe(t *testing.T) {
	var c *Capabilities
	if c.Supports("x", "temperature") || c.Known("x") {
		t.Error("nil caps must be false for everything")
	}
}

func TestNewFromMap(t *testing.T) {
	c := NewFromMap(map[string][]string{"m": {"temperature", "top_p"}})
	if !c.Supports("m", "top_p") || c.Supports("m", "frequency_penalty") {
		t.Error("NewFromMap gating wrong")
	}
}
