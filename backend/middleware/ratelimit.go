package middleware

import (
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// TokenBucket implements a thread-safe token bucket rate limiter.
type TokenBucket struct {
	mu       sync.Mutex
	tokens   float64
	capacity float64
	refillPS float64 // tokens per second
	lastTime time.Time
}

func newBucket(rpm int) *TokenBucket {
	return &TokenBucket{
		tokens:   float64(rpm),
		capacity: float64(rpm),
		refillPS: float64(rpm) / 60.0,
		lastTime: time.Now(),
	}
}

func (b *TokenBucket) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	elapsed := now.Sub(b.lastTime).Seconds()
	b.lastTime = now
	b.tokens = min64(b.capacity, b.tokens+elapsed*b.refillPS)
	if b.tokens >= 1.0 {
		b.tokens--
		return true
	}
	return false
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

var (
	globalBucket *TokenBucket
	bucketOnce   sync.Once
)

func RateLimit(rpm int) gin.HandlerFunc {
	bucketOnce.Do(func() {
		globalBucket = newBucket(rpm)
	})
	return func(c *gin.Context) {
		if !globalBucket.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error": "rate limit exceeded — max 20 req/min",
			})
			return
		}
		c.Next()
	}
}

func CORS() gin.HandlerFunc {
	// Allow local dev (localhost on any port), common free-host suffixes, and any
	// exact origins listed in CORS_ALLOWED_ORIGINS (comma-separated). The previous
	// implementation reflected any non-empty Origin, which is effectively no CORS
	// policy at all.
	extra := map[string]bool{}
	for _, o := range strings.Split(os.Getenv("CORS_ALLOWED_ORIGINS"), ",") {
		if o = strings.TrimSpace(o); o != "" {
			extra[o] = true
		}
	}
	suffixes := []string{".vercel.app", ".up.railway.app", ".onrender.com"}

	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		if origin != "" && originAllowed(origin, extra, suffixes) {
			c.Header("Access-Control-Allow-Origin", origin)
			c.Header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
			c.Header("Access-Control-Expose-Headers", "X-Route-Path, X-Hop-Count, X-Latency-Ms, X-Freq-Penalty")
		}
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

// originAllowed reports whether an incoming Origin header is permitted: an exact
// match in the configured set, any localhost/127.0.0.1 port (local dev), or a
// host ending in one of the allowed deployment suffixes.
func originAllowed(origin string, exact map[string]bool, suffixes []string) bool {
	if exact[origin] {
		return true
	}
	if strings.HasPrefix(origin, "http://localhost") || strings.HasPrefix(origin, "http://127.0.0.1") {
		return true
	}
	host := origin
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+3:]
	}
	if i := strings.IndexByte(host, '/'); i >= 0 {
		host = host[:i]
	}
	if i := strings.IndexByte(host, ':'); i >= 0 {
		host = host[:i]
	}
	for _, suf := range suffixes {
		if strings.HasSuffix(host, suf) {
			return true
		}
	}
	return false
}
