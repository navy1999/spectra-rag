package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/navy1999/spectra-rag/backend/config"
	"github.com/navy1999/spectra-rag/backend/retrieval"
)

const newGraph = `{"nodes":[{"id":"a","type":"paper","name":"Alpha"},{"id":"b","type":"topic","name":"Beta"}],"edges":[{"from":"a","to":"b","rel":"about"}]}`

func ingestRouter(token string) *gin.Engine {
	gin.SetMode(gin.TestMode)
	g, _ := retrieval.ParseGraph([]byte(`{"nodes":[{"id":"n1","type":"paper","name":"Seed"}],"edges":[]}`))
	h := New(&config.Config{IngestToken: token}, retrieval.NewStore(g), nil)
	r := gin.New()
	r.POST("/ingest", h.Ingest)
	r.GET("/graph", h.GraphInfo)
	return r
}

func post(r *gin.Engine, body, token string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/ingest", bytes.NewBufferString(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	r.ServeHTTP(w, req)
	return w
}

func TestIngest_DisabledWithoutToken(t *testing.T) {
	if w := post(ingestRouter(""), newGraph, "secret"); w.Code != http.StatusForbidden {
		t.Errorf("code = %d, want 403 when INGEST_TOKEN unset", w.Code)
	}
}

func TestIngest_WrongToken(t *testing.T) {
	if w := post(ingestRouter("secret"), newGraph, "nope"); w.Code != http.StatusUnauthorized {
		t.Errorf("code = %d, want 401", w.Code)
	}
}

func TestIngest_InvalidGraph(t *testing.T) {
	if w := post(ingestRouter("secret"), `{"nodes":[],"edges":[]}`, "secret"); w.Code != http.StatusBadRequest {
		t.Errorf("code = %d, want 400 for empty graph", w.Code)
	}
}

func TestIngest_SuccessHotSwaps(t *testing.T) {
	r := ingestRouter("secret")
	if w := post(r, newGraph, "secret"); w.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	// The active graph should now reflect the ingested one.
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/graph", nil)
	r.ServeHTTP(w, req)
	if !bytes.Contains(w.Body.Bytes(), []byte(`"nodes":2`)) {
		t.Errorf("graph did not hot-swap: %s", w.Body.String())
	}
}
