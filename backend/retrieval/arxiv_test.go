package retrieval

import "testing"

const sampleFeed = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.01234v2</id>
    <title>Efficient Retrieval over
      Knowledge Graphs</title>
    <summary>  We propose a method   for retrieval. </summary>
    <published>2025-01-15T10:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.IR"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2412.55555v1</id>
    <title>A Second Paper</title>
    <summary>Another abstract.</summary>
    <published>2024-12-01T00:00:00Z</published>
    <author><name>Grace Hopper</name></author>
    <category term="cs.LG"/>
  </entry>
</feed>`

func TestParseArxivFeed(t *testing.T) {
	papers, err := parseArxivFeed([]byte(sampleFeed))
	if err != nil {
		t.Fatal(err)
	}
	if len(papers) != 2 {
		t.Fatalf("want 2 papers, got %d", len(papers))
	}
	p := papers[0]
	if p.ID != "2501.01234" {
		t.Errorf("id = %q, want bare id without version", p.ID)
	}
	if p.Title != "Efficient Retrieval over Knowledge Graphs" {
		t.Errorf("title whitespace not collapsed: %q", p.Title)
	}
	if p.Summary != "We propose a method for retrieval." {
		t.Errorf("summary = %q", p.Summary)
	}
	if p.Year != 2025 {
		t.Errorf("year = %d, want 2025", p.Year)
	}
	if len(p.Authors) != 2 || p.Authors[0] != "Ada Lovelace" {
		t.Errorf("authors = %v", p.Authors)
	}
	if len(p.Categories) != 2 || p.Categories[0] != "cs.IR" {
		t.Errorf("categories = %v", p.Categories)
	}
}

func TestArxivIDFromURL(t *testing.T) {
	cases := map[string]string{
		"http://arxiv.org/abs/2501.01234v2": "2501.01234",
		"http://arxiv.org/abs/2412.55555":   "2412.55555",
		"2301.00001v10":                     "2301.00001",
	}
	for in, want := range cases {
		if got := arxivIDFromURL(in); got != want {
			t.Errorf("arxivIDFromURL(%q) = %q, want %q", in, got, want)
		}
	}
}
