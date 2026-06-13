package eval

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Client is a non-streaming chat client with a persistent on-disk response
// cache and 429-aware backoff. Caching is essential on free tiers: re-runs and
// the identical-prompt overlap between conditions cost zero extra calls, and an
// interrupted run resumes without repeating work.
type Client struct {
	APIKey     string
	BaseURL    string
	Model      string
	MinDelay   time.Duration // polite spacing between live calls
	MaxRetries int           // attempts per call on 429/5xx

	cachePath string
	cache     map[string]string
	http      *http.Client
}

func NewClient(apiKey, baseURL, model, cachePath string) *Client {
	c := &Client{
		APIKey:     apiKey,
		BaseURL:    baseURL,
		Model:      model,
		MinDelay:   600 * time.Millisecond,
		MaxRetries: 3,
		cachePath:  cachePath,
		cache:      map[string]string{},
		http:       &http.Client{Timeout: 90 * time.Second},
	}
	if data, err := os.ReadFile(cachePath); err == nil {
		_ = json.Unmarshal(data, &c.cache)
	}
	return c
}

// Preflight makes one minimal, un-cached, no-retry call to check whether the
// model is currently serving (free models are frequently rate-limited upstream).
// Returns nil if the provider answers, an error otherwise — so the runner can
// fail over to another candidate fast instead of grinding through every call.
func (c *Client) Preflight() error {
	body, _ := json.Marshal(map[string]interface{}{
		"model":      c.Model,
		"messages":   []map[string]string{{"role": "user", "content": "ping"}},
		"max_tokens": 1,
		"stream":     false,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(raw), 120))
	}
	return nil
}

func (c *Client) cacheKey(prompt string, temp float64, maxTokens int) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%.3f\x00%d\x00%s", c.Model, temp, maxTokens, prompt)))
	return fmt.Sprintf("%x", h[:])
}

// Complete returns the model's answer for a single user prompt. Cache hits are
// instant; misses make a live call with retry/backoff and are persisted.
func (c *Client) Complete(prompt string, temp float64, maxTokens int) (string, error) {
	key := c.cacheKey(prompt, temp, maxTokens)
	if v, ok := c.cache[key]; ok {
		return v, nil
	}

	body, _ := json.Marshal(map[string]interface{}{
		"model":       c.Model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"temperature": temp,
		"max_tokens":  maxTokens,
		"stream":      false,
	})

	var lastErr error
	for attempt := 0; attempt < c.MaxRetries; attempt++ {
		if c.MinDelay > 0 {
			time.Sleep(c.MinDelay)
		}
		req, _ := http.NewRequest("POST", c.BaseURL+"/chat/completions", bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			backoff(attempt, 0)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			retryAfter, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
			lastErr = fmt.Errorf("provider HTTP %d: %s", resp.StatusCode, truncate(string(raw), 160))
			backoff(attempt, retryAfter)
			continue
		}
		if resp.StatusCode >= 400 {
			return "", fmt.Errorf("provider HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
		}

		var parsed struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
			return "", fmt.Errorf("decode response: %w", err)
		}
		text := strings.TrimSpace(parsed.Choices[0].Message.Content)
		c.cache[key] = text
		c.save()
		return text, nil
	}
	return "", fmt.Errorf("exhausted retries: %w", lastErr)
}

func (c *Client) save() {
	if data, err := json.MarshalIndent(c.cache, "", "  "); err == nil {
		_ = os.WriteFile(c.cachePath, data, 0o644)
	}
}

func backoff(attempt, retryAfterSec int) {
	if retryAfterSec > 0 {
		time.Sleep(time.Duration(retryAfterSec) * time.Second)
		return
	}
	// 2s, 4s, 8s, 16s, ...
	time.Sleep(time.Duration(2<<attempt) * time.Second)
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
