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
}

func New(cfg *config.Config, store *retrieval.Store) *Handler {
	pcaRouter, _ := router.NewPCARouter(cfg.PCACentroidsPath)
	return &Handler{
		cfg:      cfg,
		store:    store,
		embedder: retrieval.NewEmbedder(cfg.EmbeddingsAPIKey, cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel),
		router:   pcaRouter,
	}
}

type QueryRequest struct {
	Query string `json:"query" binding:"required"`
	Model string `json:"model"` // optional per-request model override
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
		loop := agent.NewAgentLoop(evalCfg, h.store.Graph(), h.cfg.MaxHops)
		var metrics agent.AgentMetrics
		contextChunks, metrics = loop.Run(req.Query)
		hops = metrics.HopsUsed
	}

	// 4. Algorithm 3: SVD/TF-IDF redundancy penalty over the retrieved context.
	// Free models reject a frequency_penalty parameter, so the scalar is turned
	// into a natural-language synthesis directive (PenaltyInstruction) below.
	freqPenalty := synthesis.ComputeFrequencyPenalty(contextChunks)

	// 5. Route headers (known pre-stream) + in-band route event (full detail,
	// consumed by the pipeline inspector UI).
	c.Header("X-Route-Path", string(decision.Path))
	c.Header("X-Hop-Count", strconv.Itoa(hops))
	c.Header("X-Route-Regime", decision.Regime)
	c.Header("X-Route-Confidence", strconv.FormatFloat(decision.Confidence, 'f', 2, 64))
	c.Header("X-Freq-Penalty", strconv.FormatFloat(freqPenalty, 'f', 3, 64))
	routeEvt := routeEvent(decision, hops, len(contextChunks), freqPenalty, h.router.Centroids())

	// 6. Build prompt
	var sb strings.Builder
	sb.WriteString("You are a helpful research assistant with access to an academic knowledge graph.\n\n")
	if instr := synthesis.PenaltyInstruction(freqPenalty); instr != "" {
		sb.WriteString(instr + "\n\n")
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
	h.streamLLM(c, sb.String(), candidates, decision.Temperature, interceptor, routeEvt, start)
}

// dispatchLLM issues a single streaming completion request for one model and
// returns the live response. The caller decides whether to retry on the status.
func (h *Handler) dispatchLLM(prompt, model string, temp float64) (*http.Response, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": temp,
		"stream":      true,
	})
	req, _ := http.NewRequest("POST", h.cfg.OpenRouterBaseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+h.cfg.OpenRouterAPIKey)
	req.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(req)
}

// routeEvent encodes the leading SSE event describing the routing decision and
// retrieval outcome — everything the pipeline inspector visualizes. Sent before
// any token so the UI can light up stages while the answer streams.
func routeEvent(d *router.RouteDecision, hops, chunks int, freqPenalty float64, centroids []router.Centroid) string {
	cents := make([]map[string]interface{}, 0, len(centroids))
	for _, c := range centroids {
		cents = append(cents, map[string]interface{}{"name": c.Name, "x": c.X, "y": c.Y})
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
			"chunks":      chunks,
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
		for _, tok := range tokens {
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": tok + " "}))
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

func (h *Handler) streamLLM(c *gin.Context, prompt string, models []string, temp float64, interceptor *trie.StreamInterceptor, routeEvt string, start time.Time) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")

	// Walk the candidate models, retrying only on a 429 (free-model rate limit).
	// Any other error is terminal — surfaced to the diagnostics panel as-is.
	var resp *http.Response
	var lastErr error
	var lastCode int
	var lastDetail string
	for i, model := range models {
		r, err := h.dispatchLLM(prompt, model, temp)
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
			}
		}
		// Upstream closed the stream without an explicit [DONE].
		finish()
		return false
	})
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}
