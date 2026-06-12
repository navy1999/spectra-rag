package handlers

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"model":  h.cfg.DefaultModel,
		"mock":   h.cfg.MockLLM,
	})
}
