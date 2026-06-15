package retrieval

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ArxivPaper is one paper fetched from the arXiv API, normalized into the shape
// the graph builder consumes.
type ArxivPaper struct {
	ID         string // bare arXiv id, e.g. "2501.01234"
	Title      string
	Authors    []string
	Summary    string
	Year       int
	Categories []string
}

type atomFeed struct {
	Entries []atomEntry `xml:"entry"`
}

type atomEntry struct {
	ID        string `xml:"id"`
	Title     string `xml:"title"`
	Summary   string `xml:"summary"`
	Published string `xml:"published"`
	Authors   []struct {
		Name string `xml:"name"`
	} `xml:"author"`
	Categories []struct {
		Term string `xml:"term,attr"`
	} `xml:"category"`
}

// FetchArxiv queries the arXiv API for the most recent papers matching query and
// returns up to maxResults of them. baseURL defaults to the public arXiv API.
// One request covers the bounded result count this project uses (arXiv allows
// large max_results in a single call); we sort by submission date so the corpus
// is recent — i.e. authors a small model has not memorized.
func FetchArxiv(baseURL, query string, maxResults int) ([]ArxivPaper, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("empty query")
	}
	if maxResults <= 0 {
		maxResults = 25
	}
	if baseURL == "" {
		baseURL = "https://export.arxiv.org/api"
	}
	endpoint := fmt.Sprintf("%s/query?search_query=%s&start=0&max_results=%d&sortBy=submittedDate&sortOrder=descending",
		strings.TrimRight(baseURL, "/"), url.QueryEscape("all:"+query), maxResults)

	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Get(endpoint)
	if err != nil {
		return nil, fmt.Errorf("arxiv request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("arxiv HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read arxiv response: %w", err)
	}
	return parseArxivFeed(body)
}

// parseArxivFeed converts arXiv Atom XML into normalized papers. Separated from
// FetchArxiv so it is unit-testable without network.
func parseArxivFeed(body []byte) ([]ArxivPaper, error) {
	var feed atomFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return nil, fmt.Errorf("parse arxiv feed: %w", err)
	}
	papers := make([]ArxivPaper, 0, len(feed.Entries))
	for _, e := range feed.Entries {
		id := arxivIDFromURL(e.ID)
		title := collapseWS(e.Title)
		if id == "" || title == "" {
			continue
		}
		p := ArxivPaper{
			ID:      id,
			Title:   title,
			Summary: collapseWS(e.Summary),
			Year:    yearFrom(e.Published),
		}
		for _, a := range e.Authors {
			if name := collapseWS(a.Name); name != "" {
				p.Authors = append(p.Authors, name)
			}
			if len(p.Authors) >= 6 {
				break
			}
		}
		for _, c := range e.Categories {
			if c.Term != "" {
				p.Categories = append(p.Categories, c.Term)
			}
			if len(p.Categories) >= 3 {
				break
			}
		}
		papers = append(papers, p)
	}
	return papers, nil
}

// arxivIDFromURL turns "http://arxiv.org/abs/2501.01234v2" into "2501.01234".
func arxivIDFromURL(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}
	if i := strings.LastIndex(s, "v"); i > 0 { // strip version suffix
		if _, err := fmt.Sscanf(s[i+1:], "%d", new(int)); err == nil {
			s = s[:i]
		}
	}
	return s
}

func yearFrom(published string) int {
	if t, err := time.Parse(time.RFC3339, published); err == nil {
		return t.Year()
	}
	if len(published) >= 4 {
		var y int
		if _, err := fmt.Sscanf(published[:4], "%d", &y); err == nil {
			return y
		}
	}
	return 0
}

func collapseWS(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
