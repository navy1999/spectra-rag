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
	mu        sync.RWMutex
	graph     *Graph
	trie      *trie.Trie
	nodeIndex *NodeIndex // optional semantic seed index, swapped with the graph
}

// NewStore builds a Store around an initial graph, constructing its Trie.
func NewStore(g *Graph) *Store {
	s := &Store{}
	s.Set(g)
	return s
}

// Set atomically replaces the active graph, rebuilds the Trie, and clears the
// node index (a stale index from the previous graph would return ids that no
// longer exist). Use SetWithIndex when a matching index is available.
func (s *Store) Set(g *Graph) {
	s.SetWithIndex(g, nil)
}

// SetWithIndex atomically replaces the graph + Trie + semantic node index
// together — the path used by topic ingestion, which embeds the new graph's
// nodes and builds a matching index.
func (s *Store) SetWithIndex(g *Graph, idx *NodeIndex) {
	t := trie.New()
	for _, name := range g.AllNodeNames() {
		t.Insert(name)
	}
	s.mu.Lock()
	s.graph = g
	s.trie = t
	s.nodeIndex = idx
	s.mu.Unlock()
}

// SetNodeIndex attaches a node index to the current graph without swapping it —
// used at startup to register the prebuilt node_embeddings.json index.
func (s *Store) SetNodeIndex(idx *NodeIndex) {
	s.mu.Lock()
	s.nodeIndex = idx
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

// NodeIndex returns the current semantic node index snapshot (nil if none).
func (s *Store) NodeIndex() *NodeIndex {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.nodeIndex
}
