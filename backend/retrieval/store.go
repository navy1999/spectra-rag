package retrieval

import (
	"sync"

	"github.com/navy1999/spectra-rag/backend/trie"
)

// Store holds the active knowledge graph and the entity Trie derived from it,
// and lets them be hot-swapped atomically at runtime (e.g. via /ingest) while
// requests are in flight. Graph and Trie values are never mutated in place —
// a swap replaces the pointers — so a reader that takes a snapshot keeps a
// consistent view even if a swap lands mid-request.
type Store struct {
	mu    sync.RWMutex
	graph *Graph
	trie  *trie.Trie
}

// NewStore builds a Store around an initial graph, constructing its Trie.
func NewStore(g *Graph) *Store {
	s := &Store{}
	s.Set(g)
	return s
}

// Set atomically replaces the active graph and rebuilds the Trie from its node
// names.
func (s *Store) Set(g *Graph) {
	t := trie.New()
	for _, name := range g.AllNodeNames() {
		t.Insert(name)
	}
	s.mu.Lock()
	s.graph = g
	s.trie = t
	s.mu.Unlock()
}

// Graph returns the current graph snapshot.
func (s *Store) Graph() *Graph {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.graph
}

// Trie returns the current Trie snapshot.
func (s *Store) Trie() *trie.Trie {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.trie
}
