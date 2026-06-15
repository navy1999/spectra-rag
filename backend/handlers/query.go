package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/navy1999/spectra-rag/backend/agent"
	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/llmcaps"
	"github.com/navy1999/spectra-rag/backend/retrieval"
	"github.com/navy1999/spectra-rag/backend/router"
	"github.com/navy1999/spectra-rag/backend/synthesis"
	"github.com/navy1999/spectra-rag/backend/trie"
)

type Handler struct {
	cfg      *config.Config
	store    *retrieval.Store
	embedder *retrieval.Embedder
	router   *router.PCARouter
	caps     *llmcaps.Capabilities // optional: per-model supported sampling params
}

func New(cfg *config.Config, store *retrieval.Store, nodeIndex *retrieval.NodeIndex) *Handler {
	pcaRouter, _ := router.NewPCARouter(cfg.PCACentroidsPath)
	// Optional supervised path classifier (raw shrinkage-LDA, 97.5% LOO). When
	// present it overrides the PCA-2D chat/agentic decision; absent → PCA policy.
	if err := pcaRouter.LoadLDA(cfg.LDARouterPath); err != nil {
		log.Printf("[spectra-rag] LDA router not loaded (%v) — routing uses the PCA-2D policy", err)
	} else {
		log.Printf("[spectra-rag] LDA router loaded from %s — supervised chat/agentic routing active", cfg.LDARouterPath)
	}
	// Register the prebuilt node index on the Store so the agent loop reads it —
	// and any topic-ingested replacement — from a single swappable source.
	if nodeIndex != nil {
		store.SetNodeIndex(nodeIndex)
	}
	return &Handler{
		cfg:      cfg,
		store:    store,
		embedder: retrieval.NewEmbedderWithTask(cfg.EmbeddingsAPIKey, cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel, cfg.EmbeddingsTask),
		router:   pcaRouter,
	}
}

// SetCapabilities attaches the per-model sampling-parameter table (fetched from
// OpenRouter at startup). When set, the request builder sends native params the
// model supports (e.g. frequency_penalty) instead of the prompt-directive
// fallback. Nil/unset → temperature-only, which is always safe.
func (h *Handler) SetCapabilities(c *llmcaps.Capabilities) { h.caps = c }

type QueryRequest struct {
	Query         string `json:"query" binding:"required"`
	Model         string `json:"model"`          // optional per-request model override
	ForceRetrieve bool   `json:"force_retrieve"` // force the retrieval path regardless of the router
}

func (h *Handler) Query(c *gin.Context) {
	var req QueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	start := time.Now()

	// A request may override the model (UI model picker); otherwise use the
	// configured default. The same model is used for generation and for the
	// evaluator votes so the run is internally consistent.
	model := h.cfg.DefaultModel
	if m := strings.TrimSpace(req.Model); m != "" {
		model = m
	}

	// The full pipeline (embed → route → retrieve → penalty) runs even in MOCK
	// mode — the embedder falls back to a deterministic hash embedding and the
	// evaluator short-circuits, so only the final LLM stream is synthetic. This
	// keeps the pipeline visualization meaningful without an API key.

	// 1. Embed. On failure we do NOT fabricate a vector (that would make routing
	// silently meaningless); the router falls back to its default and we log it.
	emb, err := h.embedder.Embed(req.Query)
	if err != nil {
		log.Printf("[spectra-rag] embedding failed, routing degraded to default: %v", err)
		emb = nil
	}

	// 2. Route
	decision, _ := h.router.Route(emb)
	// Force the retrieval path when a custom corpus is active (you ingested it, so
	// use it) or when the caller asks explicitly. The intent router would
	// otherwise send some queries to chat and answer from model memory, silently
	// ignoring a freshly ingested graph.
	forced := h.store.Custom() || req.ForceRetrieve
	if forced {
		decision.Path = router.PathAgentic
	}

	// 3. Retrieve context (agentic path only)
	var contextChunks []string
	hops := 0
	if decision.Path == router.PathAgentic {
		evalCfg := agent.EvaluatorConfig{
			APIKey:  h.cfg.OpenRouterAPIKey,
			BaseURL: h.cfg.OpenRouterBaseURL,
			Model:   model,
			MockLLM: h.cfg.MockLLM,
		}
		loop := agent.NewAgentLoop(evalCfg, h.store.Graph(), h.store.NodeIndex(), h.cfg.MaxHops)
		var metrics agent.AgentMetrics
		contextChunks, metrics = loop.Run(req.Query, emb)
		hops = metrics.HopsUsed
	}

	// 4. Algorithm 3: SVD/TF-IDF redundancy penalty over the retrieved context.
	// If the model natively supports frequency_penalty (verified per-model via
	// llmcaps — most do, including liquid/lfm-2.5-1.2b-instruct), we send the
	// scalar as the REAL parameter; only when it does NOT do we fall back to the
	// natural-language synthesis directive below. "Use the native knob where it
	// exists; reconstruct it heuristically only when the provider omits it."
	freqPenalty := synthesis.ComputeFrequencyPenalty(contextChunks)
	useNativeFreqPenalty := h.caps.Supports(model, "frequency_penalty")

	// 5. Route headers (known pre-stream) + in-band route event (full detail,
	// consumed by the pipeline inspector UI).
	c.Header("X-Route-Path", string(decision.Path))
	c.Header("X-Hop-Count", strconv.Itoa(hops))
	c.Header("X-Route-Regime", decision.Regime)
	c.Header("X-Route-Confidence", strconv.FormatFloat(decision.Confidence, 'f', 2, 64))
	c.Header("X-Freq-Penalty", strconv.FormatFloat(freqPenalty, 'f', 3, 64))
	routeEvt := routeEvent(decision, hops, contextChunks, freqPenalty, h.router.Centroids())

	// 6. Build prompt
	var sb strings.Builder
	sb.WriteString("You are a helpful research assistant with access to an academic knowledge graph.\n\n")
	if !useNativeFreqPenalty {
		if instr := synthesis.PenaltyInstruction(freqPenalty); instr != "" {
			sb.WriteString(instr + "\n\n")
		}
	}
	if len(contextChunks) > 0 {
		sb.WriteString("Relevant context:\n")
		for _, chunk := range contextChunks {
			sb.WriteString("- " + chunk + "\n")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("User question: ")
	sb.WriteString(req.Query)

	// 7. Stream
	if h.cfg.MockLLM {
		h.streamMock(c, req.Query, model, decision, len(contextChunks), routeEvt, start)
		return
	}
	interceptor := trie.NewInterceptor(h.store.Trie())
	// Try the chosen model first, then fall through the configured free-model
	// fallbacks on a 429 so a rate-limited free model doesn't kill the request.
	candidates := append([]string{model}, h.cfg.FallbackModels...)

	// Only signal-driven knobs are sent: temperature (router: regime + novelty)
	// and frequency_penalty (A3: SVD redundancy of the retrieved context, when the
	// model supports the native param). top_p and presence_penalty are left at the
	// provider default (neutral) — we have no earned signal for them, and top_p in
	// particular is redundant with temperature. dispatchLLM drops any field a given
	// candidate model rejects.
	profile := SamplingProfile{Temperature: decision.Temperature}
	if useNativeFreqPenalty {
		fp := freqPenalty
		if fp > 1.0 {
			fp = 1.0
		}
		profile.FrequencyPenalty = &fp
	}
	h.streamLLM(c, sb.String(), candidates, profile, interceptor, routeEvt, start)
}

// dispatchLLM issues a single streaming completion request for one model and
// returns the live response. The caller decides whether to retry on the status.
func (h *Handler) dispatchLLM(prompt, model string, profile SamplingProfile) (*http.Response, error) {
	// samplingPayload sends temperature plus any other knob (top_p, presence/
	// frequency penalty) the model's provider natively accepts; unsupported ones
	// are dropped so the provider never 400s.
	payload := samplingPayload(model, profile, h.caps)
	payload["model"] = model
	payload["messages"] = []map[string]string{{"role": "user", "content": prompt}}
	payload["stream"] = true
	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", h.cfg.OpenRouterBaseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+h.cfg.OpenRouterAPIKey)
	req.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(req)
}

// routeEvent encodes the leading SSE event describing the routing decision and
// retrieval outcome — everything the pipeline inspector visualizes. Sent before
// any token so the UI can light up stages while the answer streams.
func routeEvent(d *router.RouteDecision, hops int, chunks []string, freqPenalty float64, centroids []router.Centroid) string {
	cents := make([]map[string]interface{}, 0, len(centroids))
	for _, c := range centroids {
		cents = append(cents, map[string]interface{}{"name": c.Name, "x": c.X, "y": c.Y})
	}
	// Retrieved chunk labels, truncated — so the UI can show exactly which graph
	// nodes grounded the answer (and A/B retrieval without the LLM-output confound).
	retrieved := make([]string, 0, len(chunks))
	for _, ch := range chunks {
		if len(ch) > 110 {
			ch = ch[:110] + "…"
		}
		retrieved = append(retrieved, ch)
	}
	return mustJSON(map[string]interface{}{
		"route": map[string]interface{}{
			"path":        string(d.Path),
			"regime":      d.Regime,
			"confidence":  d.Confidence,
			"temperature": d.Temperature,
			"x":           d.PCAX,
			"y":           d.PCAY,
			"distance":    d.Distance,
			"hops":        hops,
			"chunks":      len(chunks),
			"retrieved":   retrieved,
			"freqPenalty": freqPenalty,
			"centroids":   cents,
		},
	})
}

func (h *Handler) streamMock(c *gin.Context, query, model string, decision *router.RouteDecision, chunks int, routeEvt string, start time.Time) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")

	tokens := strings.Fields(
		fmt.Sprintf("Mock answer for %q. The pipeline ran for real, though: "+
			"the query embedded and projected into PCA space, landed in the %s regime "+
			"with %.0f%% confidence, took the %s path, and retrieved %d context chunk(s) "+
			"from the knowledge graph. Set OPENROUTER_API_KEY and MOCK_LLM=false to get "+
			"real answers from %s.",
			query, decision.Regime, decision.Confidence*100, decision.Path, chunks, model))

	c.Stream(func(w io.Writer) bool {
		fmt.Fprintf(w, "data: %s\n\n", routeEvt)
		flush(w)
		for _, tok := range tokens {
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": tok + " "}))
			flush(w)
			time.Sleep(30 * time.Millisecond)
		}
		latency := time.Since(start).Milliseconds()
		fmt.Fprintf(w, "data: %s\n\n", metaEvent(latency, 0))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		return false
	})
}

// errorEvent encodes a diagnostic SSE event. It is intentionally a distinct
// channel from token events so the client can surface failures in its
// diagnostics UI rather than printing operational text into the answer body.
func errorEvent(stage string, code int, message string) string {
	return mustJSON(map[string]interface{}{
		"error": map[string]interface{}{
			"stage":   stage,
			"code":    code,
			"message": message,
		},
	})
}

// summarizeProviderError turns a raw provider error body into a short, human
// sentence for the diagnostics panel. Free models are frequently rate-limited
// upstream; that case gets a clearer message than the raw JSON.
func summarizeProviderError(code int, body string) string {
	if code == 429 || strings.Contains(strings.ToLower(body), "rate-limit") {
		return "The free model is rate-limited upstream right now. Retry shortly, or set DEFAULT_MODEL to a less congested free model."
	}
	if len(body) > 200 {
		body = body[:200] + "…"
	}
	if body == "" {
		return fmt.Sprintf("Model provider returned HTTP %d.", code)
	}
	return fmt.Sprintf("Model provider returned HTTP %d: %s", code, body)
}

// metaEvent encodes the trailing SSE event carrying post-stream metrics. These
// are sent in-band (not as response headers) because their values are only known
// after streaming has begun, at which point headers are already flushed.
func metaEvent(latencyMs int64, interceptions int) string {
	return mustJSON(map[string]interface{}{
		"meta": map[string]interface{}{
			"latencyMs":     latencyMs,
			"interceptions": interceptions,
		},
	})
}

func (h *Handler) streamLLM(c *gin.Context, prompt string, models []string, profile SamplingProfile, interceptor *trie.StreamInterceptor, routeEvt string, start time.Time) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")

	// Walk the candidate models, retrying only on a 429 (free-model rate limit).
	// Any other error is terminal — surfaced to the diagnostics panel as-is.
	var resp *http.Response
	var lastErr error
	var lastCode int
	var lastDetail string
	for i, model := range models {
		r, err := h.dispatchLLM(prompt, model, profile)
		if err != nil {
			lastErr, lastCode = err, 0
			continue
		}
		if r.StatusCode == http.StatusTooManyRequests {
			b, _ := io.ReadAll(r.Body)
			r.Body.Close()
			lastCode, lastDetail = r.StatusCode, strings.TrimSpace(string(b))
			log.Printf("[spectra-rag] model %s rate-limited (429); trying next fallback (%d left)", model, len(models)-i-1)
			continue
		}
		if r.StatusCode >= 400 {
			b, _ := io.ReadAll(r.Body)
			r.Body.Close()
			lastCode, lastDetail = r.StatusCode, strings.TrimSpace(string(b))
			log.Printf("[spectra-rag] openrouter error %d on %s: %s", r.StatusCode, model, lastDetail)
			break
		}
		resp = r
		break
	}

	if resp == nil {
		code, msg := lastCode, ""
		if lastErr != nil {
			msg = "could not reach the model provider: " + lastErr.Error()
		} else {
			msg = summarizeProviderError(lastCode, lastDetail)
		}
		c.Stream(func(w io.Writer) bool {
			fmt.Fprintf(w, "data: %s\n\n", routeEvt)
			fmt.Fprintf(w, "data: %s\n\n", errorEvent("openrouter", code, msg))
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return false
		})
		return
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	wroteRoute := false
	c.Stream(func(w io.Writer) bool {
		if !wroteRoute {
			fmt.Fprintf(w, "data: %s\n\n", routeEvt)
			flush(w)
			wroteRoute = true
		}
		finish := func() {
			if tail := interceptor.Flush(); tail != "" {
				fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": tail}))
			}
			latency := time.Since(start).Milliseconds()
			fmt.Fprintf(w, "data: %s\n\n", metaEvent(latency, interceptor.Count()))
			fmt.Fprintf(w, "data: [DONE]\n\n")
		}
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				finish()
				return false
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			token := chunk.Choices[0].Delta.Content
			if token == "" {
				continue
			}
			emit, _ := interceptor.ProcessToken(token)
			if emit != "" {
				fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": emit}))
				flush(w)
			}
		}
		// Upstream closed the stream without an explicit [DONE].
		finish()
		return false
	})
}

// flush pushes buffered SSE bytes to the client immediately. Without this,
// Go's http.ResponseWriter buffers writes (~2KB) and Gin's c.Stream only
// flushes once the step function returns, so short answers would otherwise
// arrive in one or two bursts instead of token-by-token.
func flush(w io.Writer) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}
