package router

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	pcacgo "github.com/navy1999/spectra-rag/backend/cgo"
)

type RoutePath string

const (
	PathChat    RoutePath = "chat"
	PathAgentic RoutePath = "agentic"
)

// RouteDecision carries the two independent signals the router extracts from
// the PCA projection, plus the policy outputs derived from them:
//
//   - Regime (argmin): WHICH centroid is nearest — what kind of query this is.
//     Drives the base sampling temperature.
//   - Confidence (min-distance): HOW CLOSE the query sits to known territory.
//     Low confidence = novel/out-of-distribution — drives the retrieval path
//     (agentic multi-hop) and a temperature boost.
//
// Collapsing these into one scalar (as an earlier version did) discards the
// regime label entirely; keeping both lets a "creative but familiar" query
// stream directly at high temperature while a "logic but novel" query
// triggers retrieval at low temperature.
type RouteDecision struct {
	Path        RoutePath
	Regime      string  // name of the nearest centroid (e.g. "logic", "creative")
	Confidence  float64 // 1 = at a known cluster, 0 = fully out-of-distribution
	Temperature float64
	PCAX        float64
	PCAY        float64
	Distance    float64 // raw distance to the nearest centroid
}

type Centroid struct {
	Name string  `json:"name"`
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
}

type PCARouter struct {
	centroids []Centroid
}

// Policy constants. Distances at/below distNear count as fully familiar;
// at/above distFar as fully novel. The agentic path triggers when confidence
// drops below 0.5 — geometrically the same d≈0.65 boundary the previous
// scalar-only implementation used, so routing behavior stays continuous.
const (
	distNear                = 0.3
	distFar                 = 1.0
	noveltyTempBoost        = 0.3 // added to base temp as novelty goes 0→1
	maxTemp                 = 0.9
	agenticConfidenceCutoff = 0.5
	defaultRegimeBaseTemp   = 0.35
)

// regimeBaseTemp maps a regime label to its base sampling temperature: factual
// "logic" queries answer best near-deterministic, "creative" ones benefit from
// looser sampling. Centroid names are user-defined (they come from the
// centroids JSON), so unknown labels fall back to a middle value.
var regimeBaseTemp = map[string]float64{
	"logic":    0.1,
	"creative": 0.6,
}

func NewPCARouter(centroidsPath string) (*PCARouter, error) {
	data, err := os.ReadFile(centroidsPath)
	if err != nil {
		// Use hardcoded defaults if file missing
		return &PCARouter{centroids: []Centroid{
			{Name: "logic", X: 0.42, Y: -0.18},
			{Name: "creative", X: -0.31, Y: 0.29},
		}}, nil
	}
	var c map[string][2]float64
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse centroids: %w", err)
	}
	var cents []Centroid
	for name, coords := range c {
		cents = append(cents, Centroid{Name: name, X: coords[0], Y: coords[1]})
	}
	return &PCARouter{centroids: cents}, nil
}

// Centroids returns the router's centroid set (for diagnostics/visualization).
func (r *PCARouter) Centroids() []Centroid {
	return r.centroids
}

func (r *PCARouter) Route(embedding []float32) (*RouteDecision, error) {
	proj, err := pcacgo.ProjectToPCA(embedding)
	if err != nil {
		proj = [2]float64{0, 0}
	}
	return r.decide(proj), nil
}

// decide applies the routing policy to a projected point. Split from Route so
// the policy is testable with synthetic projections, independent of which PCA
// implementation (Eigen or fallback) produced the point.
func (r *PCARouter) decide(proj [2]float64) *RouteDecision {
	regime := ""
	minDist := math.MaxFloat64
	for _, c := range r.centroids {
		if d := l2(proj[0], proj[1], c.X, c.Y); d < minDist {
			minDist = d
			regime = c.Name
		}
	}

	novelty := noveltyFromDist(minDist)
	confidence := 1 - novelty

	base, ok := regimeBaseTemp[regime]
	if !ok {
		base = defaultRegimeBaseTemp
	}
	temp := math.Min(maxTemp, base+novelty*noveltyTempBoost)

	path := PathChat
	if confidence < agenticConfidenceCutoff {
		path = PathAgentic
	}

	return &RouteDecision{
		Path:        path,
		Regime:      regime,
		Confidence:  confidence,
		Temperature: temp,
		PCAX:        proj[0],
		PCAY:        proj[1],
		Distance:    minDist,
	}
}

// noveltyFromDist maps distance-to-nearest-centroid to [0,1].
func noveltyFromDist(d float64) float64 {
	if d <= distNear {
		return 0
	}
	if d >= distFar {
		return 1
	}
	return (d - distNear) / (distFar - distNear)
}

func l2(x1, y1, x2, y2 float64) float64 {
	dx, dy := x1-x2, y1-y2
	return math.Sqrt(dx*dx + dy*dy)
}
