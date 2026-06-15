package handlers

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/retrieval"
)

type topicStatus struct {
	State      string `json:"state"` // idle | running | done | error
	Topic      string `json:"topic,omitempty"`
	Stage      string `json:"stage,omitempty"`
	Papers     int    `json:"papers,omitempty"`
	Nodes      int    `json:"nodes,omitempty"`
	Edges      int    `json:"edges,omitempty"`
	IndexDim   int    `json:"index_dim,omitempty"`  // node-embedding dim (compressed k or full)
	Compressed bool   `json:"compressed,omitempty"` // whether PCA compression engaged
	Error      string `json:"error,omitempty"`
	UpdatedAt  string `json:"updated_at,omitempty"`
}

// TopicIngester builds a knowledge graph on demand from an arXiv topic query:
// fetch (bounded) → build graph → embed nodes → atomically hot-swap the graph +
// semantic index into the Store. One ingestion runs at a time (single-slot lock);
// a second request while one is in flight is rejected with 409. State is
// in-memory and ephemeral (lost on restart) — intentional for a single-tenant
// demo on a free-tier host where RAM, arXiv latency, and embedding quota are the
// binding constraints.
type TopicIngester struct {
	store         *retrieval.Store
	embedder      *retrieval.Embedder
	arxivURL      string
	maxPapers     int
	enabled       bool
	compressDim   int // PCA target dim for large graphs (0 = never compress)
	compressAbove int // compress only when node count exceeds this

	mu sync.Mutex
	st topicStatus
}

func NewTopicIngester(cfg *config.Config, store *retrieval.Store) *TopicIngester {
	return &TopicIngester{
		store:         store,
		embedder:      retrieval.NewEmbedderWithTask(cfg.EmbeddingsAPIKey, cfg.EmbeddingsBaseURL, cfg.EmbeddingsModel, cfg.EmbeddingsTask),
		arxivURL:      cfg.ArxivBaseURL,
		maxPapers:     cfg.TopicIngestMaxPapers,
		enabled:       cfg.TopicIngestEnabled,
		compressDim:   cfg.NodeIndexCompressDim,
		compressAbove: cfg.NodeIndexCompressThreshold,
		st:            topicStatus{State: "idle"},
	}
}

type topicRequest struct {
	Query     string `json:"query" binding:"required"`
	MaxPapers int    `json:"max_papers"`
}

// Ingest starts a background topic-ingestion job (POST /ingest/topic).
func (ti *TopicIngester) Ingest(c *gin.Context) {
	if !ti.enabled {
		c.JSON(http.StatusForbidden, gin.H{"error": "topic ingestion disabled; set TOPIC_INGEST_ENABLED=true"})
		return
	}
	var req topicRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	max := ti.maxPapers
	if req.MaxPapers > 0 && req.MaxPapers < max {
		max = req.MaxPapers
	}
	if !ti.tryStart(req.Query) {
		c.JSON(http.StatusConflict, gin.H{"error": "an ingestion is already running", "status": ti.Snapshot()})
		return
	}
	go ti.run(req.Query, max)
	c.JSON(http.StatusAccepted, gin.H{"status": "started", "topic": req.Query, "max_papers": max})
}

// Status reports the current/last ingestion job (GET /ingest/status).
func (ti *TopicIngester) Status(c *gin.Context) {
	c.JSON(http.StatusOK, ti.Snapshot())
}

// --- state machine (mutex-guarded; unit-testable without network) ---

func (ti *TopicIngester) tryStart(topic string) bool {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	if ti.st.State == "running" {
		return false
	}
	ti.st = topicStatus{State: "running", Topic: topic, Stage: "queued", UpdatedAt: tsNow()}
	return true
}

func (ti *TopicIngester) setStage(stage string) {
	ti.mu.Lock()
	ti.st.Stage = stage
	ti.st.UpdatedAt = tsNow()
	ti.mu.Unlock()
}

func (ti *TopicIngester) fail(err error) {
	ti.mu.Lock()
	ti.st.State = "error"
	ti.st.Error = err.Error()
	ti.st.UpdatedAt = tsNow()
	ti.mu.Unlock()
}

func (ti *TopicIngester) done(papers, nodes, edges, indexDim int, compressed bool) {
	ti.mu.Lock()
	ti.st.State = "done"
	ti.st.Stage = "complete"
	ti.st.Papers = papers
	ti.st.Nodes = nodes
	ti.st.Edges = edges
	ti.st.IndexDim = indexDim
	ti.st.Compressed = compressed
	ti.st.UpdatedAt = tsNow()
	ti.mu.Unlock()
}

// Snapshot returns a copy of the current status (safe for concurrent reads).
func (ti *TopicIngester) Snapshot() topicStatus {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	return ti.st
}

func (ti *TopicIngester) run(query string, max int) {
	ti.setStage("fetching arXiv")
	papers, err := retrieval.FetchArxiv(ti.arxivURL, query, max)
	if err != nil {
		ti.fail(fmt.Errorf("arxiv fetch: %w", err))
		return
	}
	if len(papers) == 0 {
		ti.fail(fmt.Errorf("no papers found for %q", query))
		return
	}

	ti.setStage("building graph")
	g := retrieval.BuildGraphFromPapers(papers)
	nodes, edges, _ := g.Stats()
	if nodes == 0 {
		ti.fail(fmt.Errorf("graph build produced no nodes"))
		return
	}

	ti.setStage("embedding nodes")
	ids, texts := g.NodeTexts()
	vecs, err := ti.embedder.EmbedBatch(texts)
	if err != nil {
		ti.fail(fmt.Errorf("embed nodes: %w", err))
		return
	}

	// Size-gated PCA compression: only worth it for large graphs (the curve in
	// data/compression_curve.md). Small corpora stay full-dim for best recall.
	var idx *retrieval.NodeIndex
	if ti.compressDim > 0 && len(ids) > ti.compressAbove {
		ti.setStage(fmt.Sprintf("compressing index to %dd", ti.compressDim))
		idx = retrieval.NewCompressedNodeIndex(ids, vecs, ti.compressDim)
	} else {
		idx = retrieval.NewNodeIndex(ids, vecs)
	}

	ti.setStage("swapping in")
	ti.store.SetWithIndex(g, idx)
	ti.done(len(papers), nodes, edges, idx.Dim(), idx.Compressed())
}

func tsNow() string { return time.Now().UTC().Format(time.RFC3339) }
