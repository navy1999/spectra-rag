package handlers

import (
	"crypto/subtle"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/navy1999/spectra-rag/backend/retrieval"
)

const maxIngestBytes = 8 << 20 // 8 MiB

// Ingest replaces the active knowledge graph at runtime with a user-supplied
// graph JSON and atomically hot-swaps it (and the derived entity Trie) into the
// live Store. Gated behind a bearer token and disabled unless INGEST_TOKEN is
// set, so a public deployment is safe by default.
func (h *Handler) Ingest(c *gin.Context) {
	if h.cfg.IngestToken == "" {
		c.JSON(http.StatusForbidden, gin.H{"error": "ingestion disabled; set INGEST_TOKEN to enable"})
		return
	}
	if !validBearer(c.GetHeader("Authorization"), h.cfg.IngestToken) {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid or missing bearer token"})
		return
	}

	data, err := io.ReadAll(io.LimitReader(c.Request.Body, maxIngestBytes+1))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	if len(data) > maxIngestBytes {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": fmt.Sprintf("graph exceeds %d bytes", maxIngestBytes)})
		return
	}

	g, err := retrieval.ParseGraph(data)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	h.store.Set(g)

	nodes, edges, byType := g.Stats()
	c.JSON(http.StatusOK, gin.H{
		"status": "graph replaced",
		"nodes":  nodes,
		"edges":  edges,
		"types":  byType,
	})
}

// GraphInfo reports stats about the currently active graph.
func (h *Handler) GraphInfo(c *gin.Context) {
	nodes, edges, byType := h.store.Graph().Stats()
	c.JSON(http.StatusOK, gin.H{
		"nodes": nodes,
		"edges": edges,
		"types": byType,
	})
}

func validBearer(header, token string) bool {
	const prefix = "Bearer "
	if !strings.HasPrefix(header, prefix) {
		return false
	}
	got := strings.TrimPrefix(header, prefix)
	return subtle.ConstantTimeCompare([]byte(got), []byte(token)) == 1
}
