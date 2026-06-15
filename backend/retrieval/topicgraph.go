package retrieval

import (
	"fmt"
	"strings"
)

// topicKeywords / arxivCategoryTopics mirror scripts/build_graph.py so a graph
// built live (topic ingestion) matches one built by the offline Python pipeline.
var topicKeywords = map[string][]string{
	"Transformer Architecture":        {"transformer", "attention", "self-attention", "multi-head"},
	"Pre-training":                    {"pre-train", "pretraining", "masked language", "next sentence"},
	"Parameter-Efficient Fine-Tuning": {"lora", "adapter", "fine-tun", "peft"},
	"Efficient Attention":             {"flash", "efficient attention", "linear attention", "sparse attention"},
	"Reinforcement Learning":          {"rlhf", "reward model", "reinforcement", "human feedback"},
	"Retrieval Augmented Generation":  {"retrieval", "rag", "knowledge graph", "vector"},
	"Mixture of Experts":              {"mixture of experts", "moe", "sparse mixture"},
	"In-Context Learning":             {"in-context", "few-shot", "zero-shot", "prompt"},
}

var arxivCategoryTopics = map[string]string{
	"cs.CL":   "Computation and Language",
	"cs.LG":   "Machine Learning",
	"cs.CV":   "Computer Vision",
	"cs.AI":   "Artificial Intelligence",
	"cs.IR":   "Information Retrieval",
	"stat.ML": "Statistical Machine Learning",
	"cs.NE":   "Neural and Evolutionary Computing",
	"eess.AS": "Audio and Speech Processing",
	"cs.RO":   "Robotics",
	"cs.SD":   "Sound",
}

const maxAbstractChars = 400

// BuildGraphFromPapers constructs a knowledge graph (paper/author/topic nodes +
// authored/about edges) from fetched arXiv papers — the Go port of
// scripts/build_graph.py, for live topic ingestion. Paper abstracts are carried
// in props so retrieved chunks contain real text.
func BuildGraphFromPapers(papers []ArxivPaper) *Graph {
	var nodes []Node
	var edges []Edge
	seenPapers := map[string]bool{}
	seenAuthors := map[string]string{}
	seenTopics := map[string]string{}

	ensureTopic := func(name string) string {
		if id, ok := seenTopics[name]; ok {
			return id
		}
		id := fmt.Sprintf("t_%d", len(seenTopics))
		seenTopics[name] = id
		nodes = append(nodes, Node{ID: id, Type: NodeTopic, Name: name})
		return id
	}

	for _, p := range papers {
		if p.ID == "" || p.Title == "" || seenPapers[p.ID] {
			continue
		}
		seenPapers[p.ID] = true
		pid := "p_" + strings.ReplaceAll(p.ID, ".", "_")
		abstract := p.Summary
		if len(abstract) > maxAbstractChars {
			abstract = abstract[:maxAbstractChars]
		}
		nodes = append(nodes, Node{
			ID:    pid,
			Type:  NodePaper,
			Name:  p.Title,
			Props: map[string]interface{}{"year": p.Year, "arxiv": p.ID, "abstract": abstract},
		})

		for _, author := range p.Authors {
			aid, ok := seenAuthors[author]
			if !ok {
				aid = fmt.Sprintf("a_%d", len(seenAuthors))
				seenAuthors[author] = aid
				nodes = append(nodes, Node{ID: aid, Type: NodeAuthor, Name: author})
			}
			edges = append(edges, Edge{From: aid, To: pid, Rel: "authored"})
		}

		text := strings.ToLower(p.Title + " " + p.Summary)
		for topic, kws := range topicKeywords {
			for _, kw := range kws {
				if strings.Contains(text, kw) {
					edges = append(edges, Edge{From: pid, To: ensureTopic(topic), Rel: "about"})
					break
				}
			}
		}
		if len(p.Categories) > 0 {
			if name, ok := arxivCategoryTopics[p.Categories[0]]; ok {
				edges = append(edges, Edge{From: pid, To: ensureTopic(name), Rel: "about"})
			}
		}
	}

	return buildGraph(nodes, edges)
}

// NodeTexts returns each node's id paired with the text used to embed it
// (name + abstract), in a stable order — the input to building a NodeIndex after
// topic ingestion. Mirrors scripts/embed_nodes.py's node text.
func (g *Graph) NodeTexts() (ids []string, texts []string) {
	for id, n := range g.nodes {
		text := n.Name
		if abs, ok := n.Props["abstract"].(string); ok && abs != "" {
			text += ". " + abs
		}
		ids = append(ids, id)
		texts = append(texts, text)
	}
	return ids, texts
}
