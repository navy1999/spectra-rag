package handlers

import "github.com/navy1999/spectra-rag/backend/llmcaps"

// SamplingProfile is the full set of sampling controls the router + synthesis
// layer would like to apply to a generation. Only the parameters the target
// model actually accepts (per llmcaps.Capabilities) are sent; the rest are
// dropped so a provider never 400s on an unsupported field. Temperature is
// universal and always sent.
//
// This is the "real control surface" layer: where the model exposes the native
// knob we set it directly (e.g. frequency_penalty from the SVD redundancy
// signal, A3), instead of nudging the model in natural language.
type SamplingProfile struct {
	Temperature      float64
	TopP             *float64
	PresencePenalty  *float64
	FrequencyPenalty *float64
}

func f64(v float64) *float64 { return &v }

// samplingPayload returns the sampling fields to send for a model, dropping any
// the model's provider does not accept (per caps). A nil caps (fetch failed /
// unknown model) yields temperature-only — the safe fallback.
func samplingPayload(model string, p SamplingProfile, caps *llmcaps.Capabilities) map[string]interface{} {
	out := map[string]interface{}{"temperature": p.Temperature}
	if p.TopP != nil && caps.Supports(model, "top_p") {
		out["top_p"] = *p.TopP
	}
	if p.PresencePenalty != nil && caps.Supports(model, "presence_penalty") {
		out["presence_penalty"] = *p.PresencePenalty
	}
	if p.FrequencyPenalty != nil && caps.Supports(model, "frequency_penalty") {
		out["frequency_penalty"] = *p.FrequencyPenalty
	}
	return out
}
