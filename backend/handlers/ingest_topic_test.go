package handlers

import (
	"fmt"
	"testing"
)

func TestTopicIngester_SingleSlot(t *testing.T) {
	ti := &TopicIngester{st: topicStatus{State: "idle"}}

	if !ti.tryStart("diffusion models") {
		t.Fatal("first start should succeed")
	}
	if ti.tryStart("something else") {
		t.Error("a second start while one is running must be rejected")
	}
	if s := ti.Snapshot(); s.State != "running" || s.Topic != "diffusion models" {
		t.Errorf("running status = %+v", s)
	}

	ti.done(20, 95, 110)
	if s := ti.Snapshot(); s.State != "done" || s.Papers != 20 || s.Nodes != 95 {
		t.Errorf("done status = %+v", s)
	}

	// After completion, a new job may start.
	if !ti.tryStart("graph neural networks") {
		t.Error("start after done should succeed")
	}
	ti.fail(fmt.Errorf("boom"))
	if s := ti.Snapshot(); s.State != "error" || s.Error != "boom" {
		t.Errorf("error status = %+v", s)
	}
	if !ti.tryStart("retry") {
		t.Error("start after error should succeed")
	}
}
