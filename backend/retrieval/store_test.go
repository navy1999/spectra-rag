package retrieval

import (
	"sync"
	"testing"
)

func TestStoreSwap(t *testing.T) {
	s := NewStore(testGraph(t))
	if s.Graph().NodeCount() != 4 {
		t.Fatalf("initial nodes = %d, want 4", s.Graph().NodeCount())
	}
	if s.Trie() == nil {
		t.Fatal("trie should not be nil")
	}

	g2, err := ParseGraph([]byte(`{"nodes":[{"id":"x","type":"paper","name":"Solo Paper"}],"edges":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	s.Set(g2)
	if s.Graph().NodeCount() != 1 {
		t.Errorf("after swap nodes = %d, want 1", s.Graph().NodeCount())
	}
	if !s.Trie().SearchPrefix("Solo") {
		t.Error("trie was not rebuilt for the swapped-in graph")
	}
}

// TestStoreConcurrentReadWrite exercises hot-swap under concurrent reads; run
// with -race (as CI does) it asserts the swap is data-race free.
func TestStoreConcurrentReadWrite(t *testing.T) {
	g1 := testGraph(t)
	g2, _ := ParseGraph([]byte(`{"nodes":[{"id":"x","type":"paper","name":"Solo"}],"edges":[]}`))
	s := NewStore(g1)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = s.Graph().NodeCount()
			_ = s.Trie().AllWords()
		}()
	}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			if k%2 == 0 {
				s.Set(g1)
			} else {
				s.Set(g2)
			}
		}(i)
	}
	wg.Wait()
}
