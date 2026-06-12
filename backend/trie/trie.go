package trie

import "strings"

type TrieNode struct {
	children map[rune]*TrieNode
	isEnd    bool
	word     string
}

type Trie struct {
	root *TrieNode
}

func New() *Trie {
	return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

func (t *Trie) Insert(word string) {
	node := t.root
	lower := strings.ToLower(word)
	for _, ch := range lower {
		if node.children[ch] == nil {
			node.children[ch] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		node = node.children[ch]
	}
	node.isEnd = true
	node.word = word
}

// SearchPrefix returns true if the prefix matches any inserted word path.
func (t *Trie) SearchPrefix(prefix string) bool {
	node := t.root
	for _, ch := range strings.ToLower(prefix) {
		if node.children[ch] == nil {
			return false
		}
		node = node.children[ch]
	}
	return true
}

// ValidateAndComplete returns the best known completion for a partial token,
// or ("", false) if no completion exists.
func (t *Trie) ValidateAndComplete(partial string) (string, bool) {
	node := t.root
	for _, ch := range strings.ToLower(partial) {
		if node.children[ch] == nil {
			return "", false
		}
		node = node.children[ch]
	}
	if node.isEnd {
		return node.word, true
	}
	// DFS to find first leaf
	if w := firstWord(node); w != "" {
		return w, true
	}
	return "", false
}

func firstWord(node *TrieNode) string {
	if node.isEnd {
		return node.word
	}
	for _, child := range node.children {
		if w := firstWord(child); w != "" {
			return w
		}
	}
	return ""
}

// AllWords returns every word in the trie.
func (t *Trie) AllWords() []string {
	var words []string
	collectWords(t.root, &words)
	return words
}

func collectWords(node *TrieNode, words *[]string) {
	if node.isEnd {
		*words = append(*words, node.word)
	}
	for _, child := range node.children {
		collectWords(child, words)
	}
}
