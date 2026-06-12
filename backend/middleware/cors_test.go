package middleware

import "testing"

func TestOriginAllowed(t *testing.T) {
	suffixes := []string{".vercel.app", ".up.railway.app", ".onrender.com"}
	extra := map[string]bool{"https://my-portfolio.com": true}

	cases := []struct {
		origin string
		want   bool
	}{
		{"http://localhost:3000", true},
		{"http://localhost:5173", true},
		{"http://127.0.0.1:8080", true},
		{"https://spectra-rag.vercel.app", true},
		{"https://spectra-rag.up.railway.app", true},
		{"https://spectra.onrender.com", true},
		{"https://my-portfolio.com", true}, // exact match from configured set
		{"https://evil.com", false},
		{"https://notvercel.app.evil.com", false}, // suffix must match the host tail
		{"", false},
	}
	for _, c := range cases {
		if got := originAllowed(c.origin, extra, suffixes); got != c.want {
			t.Errorf("originAllowed(%q) = %v, want %v", c.origin, got, c.want)
		}
	}
}
