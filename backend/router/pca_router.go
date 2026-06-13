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
	// distNear/distFar bound the distance→novelty ramp. They are read from the
	// centroids file (calibrated by fit_pca to the corpus distance distribution)
	// because the right scale depends entirely on the embedding model and PCA
	// fit — the old fixed defaults were tuned for a unit-scale dev projection and
	// collapsed the router to always-chat on real embeddings.
	distNear float64
	distFar  float64
	// routeByRegime switches the chat-vs-agentic decision from the novelty/
	// confidence ramp to the nearest-class-centroid label. This is opt-in via
	// the centroids file ("route_by_regime": true) and is meant for a SUPERVISED
	// projection (LDA): when the two centroids are the chat and agentic class
	// means, "which centroid is nearest" IS the routing decision, and is far more
	// discriminative than distance-to-nearest (which is small for BOTH classes
	// near their own mean). The unsupervised PCA path leaves this false and keeps
	// the original novelty-based behavior unchanged.
	routeByRegime bool
	// chatRegime is the centroid name that maps to the chat route when
	// routeByRegime is on (default "logic"); any other nearest centroid routes
	// agentic. Lets the fitter name classes without a Go change.
	chatRegime string
}

// Policy constants. The agentic path triggers when confidence drops below 0.5.
// distNear/distFar default here but are normally overridden by the fitted file.
const (
	defaultDistNear         = 0.3
	defaultDistFar          = 1.0
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
	r := &PCARouter{distNear: defaultDistNear, distFar: defaultDistFar}

	data, err := os.ReadFile(centroidsPath)
	if err != nil {
		// Hardcoded defaults if the file is missing.
		r.centroids = []Centroid{
			{Name: "logic", X: 0.42, Y: -0.18},
			{Name: "creative", X: -0.31, Y: 0.29},
		}
		return r, nil
	}

	// Current schema: {"centroids": {name:[x,y]}, "dist_near": f, "dist_far": f}.
	var doc struct {
		Centroids     map[string][2]float64 `json:"centroids"`
		DistNear      *float64              `json:"dist_near"`
		DistFar       *float64              `json:"dist_far"`
		RouteByRegime bool                  `json:"route_by_regime"`
		ChatRegime    string                `json:"chat_regime"`
	}
	if err := json.Unmarshal(data, &doc); err == nil && len(doc.Centroids) > 0 {
		for name, xy := range doc.Centroids {
			r.centroids = append(r.centroids, Centroid{Name: name, X: xy[0], Y: xy[1]})
		}
		if doc.DistNear != nil {
			r.distNear = *doc.DistNear
		}
		if doc.DistFar != nil {
			r.distFar = *doc.DistFar
		}
		if r.distFar <= r.distNear { // guard against a degenerate ramp
			r.distNear, r.distFar = defaultDistNear, defaultDistFar
		}
		r.routeByRegime = doc.RouteByRegime
		r.chatRegime = doc.ChatRegime
		if r.chatRegime == "" {
			r.chatRegime = "logic"
		}
		return r, nil
	}

	// Legacy schema: a flat {name: [x,y]} map.
	var flat map[string][2]float64
	if err := json.Unmarshal(data, &flat); err != nil {
		return nil, fmt.Errorf("parse centroids: %w", err)
	}
	for name, xy := range flat {
		r.centroids = append(r.centroids, Centroid{Name: name, X: xy[0], Y: xy[1]})
	}
	return r, nil
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

	novelty := noveltyFromDist(minDist, r.distNear, r.distFar)
	confidence := 1 - novelty

	base, ok := regimeBaseTemp[regime]
	if !ok {
		base = defaultRegimeBaseTemp
	}
	temp := math.Min(maxTemp, base+novelty*noveltyTempBoost)

	// Routing decision. Two modes:
	//  - routeByRegime (supervised LDA): the nearest class centroid IS the route.
	//    Nearest to the chat-class centroid → chat; anything else → agentic.
	//  - default (unsupervised PCA): novelty/confidence ramp — a query that sits
	//    far from ALL known clusters is treated as out-of-distribution and sent
	//    down the agentic multi-hop path.
	path := PathChat
	if r.routeByRegime {
		if regime != r.chatRegime {
			path = PathAgentic
		}
	} else if confidence < agenticConfidenceCutoff {
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

// noveltyFromDist maps distance-to-nearest-centroid to [0,1] over the calibrated
// [near, far] ramp.
func noveltyFromDist(d, near, far float64) float64 {
	if far <= near {
		return 0
	}
	if d <= near {
		return 0
	}
	if d >= far {
		return 1
	}
	return (d - near) / (far - near)
}

func l2(x1, y1, x2, y2 float64) float64 {
	dx, dy := x1-x2, y1-y2
	return math.Sqrt(dx*dx + dy*dy)
}
