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
		embedder: retrieval.NewEmbedder(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL),
		router:   pcaRouter,
	}
}

type QueryRequest struct {
	Query string `json:"query" binding:"required"`
}

func (h *Handler) Query(c *gin.Context) {
	var req QueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	start := time.Now()

	// The full pipeline (embed → route → retrieve → penalty) runs even in MOCK
	// mode — the embedder falls back to a deterministic hash embedding and the
	// evaluator short-circuits, so only the final LLM stream is synthetic. This
	// keeps the pipeline visualization meaningful without an API key.

	// 1. Embed
	emb, err := h.embedder.Embed(req.Query)
	if err != nil {
		emb = make([]float32, 384)
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
			Model:   h.cfg.DefaultModel,
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
		h.streamMock(c, req.Query, decision, len(contextChunks), routeEvt, start)
		return
	}
	interceptor := trie.NewInterceptor(h.store.Trie())
	h.streamLLM(c, sb.String(), decision.Temperature, interceptor, routeEvt, start)
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

func (h *Handler) streamMock(c *gin.Context, query string, decision *router.RouteDecision, chunks int, routeEvt string, start time.Time) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")

	tokens := strings.Fields(
		fmt.Sprintf("Mock answer for %q. The pipeline ran for real, though: "+
			"the query embedded and projected into PCA space, landed in the %s regime "+
			"with %.0f%% confidence, took the %s path, and retrieved %d context chunk(s) "+
			"from the knowledge graph. Set OPENROUTER_API_KEY and MOCK_LLM=false to get "+
			"real answers from %s.",
			query, decision.Regime, decision.Confidence*100, decision.Path, chunks, h.cfg.DefaultModel))

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

func (h *Handler) streamLLM(c *gin.Context, prompt string, temp float64, interceptor *trie.StreamInterceptor, routeEvt string, start time.Time) {
	body, _ := json.Marshal(map[string]interface{}{
		"model": h.cfg.DefaultModel,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": temp,
		"stream":      true,
	})

	req, _ := http.NewRequest("POST", h.cfg.OpenRouterBaseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+h.cfg.OpenRouterAPIKey)
	req.Header.Set("Content-Type", "application/json")

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.Stream(func(w io.Writer) bool {
			fmt.Fprintf(w, "data: %s\n\n", routeEvt)
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": "[Error contacting OpenRouter: " + err.Error() + "]"}))
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return false
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("[spectra-rag] openrouter error %d: %s", resp.StatusCode, strings.TrimSpace(string(errBody)))
		msg := fmt.Sprintf("[OpenRouter API error %d: %s]", resp.StatusCode, strings.TrimSpace(string(errBody)))
		c.Stream(func(w io.Writer) bool {
			fmt.Fprintf(w, "data: %s\n\n", routeEvt)
			fmt.Fprintf(w, "data: %s\n\n", mustJSON(map[string]string{"token": msg}))
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return false
		})
		return
	}

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
