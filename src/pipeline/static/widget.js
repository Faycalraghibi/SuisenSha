/**
 * SuisenSha Embeddable Recommendation Carousel Widget
 * Provides real-time recommendation carousels, explainability badges,
 * DSA compliance transparency modals, and click telemetry.
 */
(function (global) {
  "use strict";

  class SuisenShaWidget {
    constructor(config) {
      this.config = Object.assign(
        {
          target: "#suisensha-carousel",
          apiUrl: window.location.origin,
          apiKey: "",
          tenantId: "default",
          userId: null,
          sessionId: null,
          endpoint: "/v1/recommend/popular",
          title: "Recommended for You",
          k: 10,
          onItemClick: null,
          onExclude: null,
        },
        config
      );

      this.container =
        typeof this.config.target === "string"
          ? document.querySelector(this.config.target)
          : this.config.target;

      this.items = [];
      this.init();
    }

    async init() {
      if (!this.container) {
        console.error("SuisenShaWidget: target container not found.");
        return;
      }
      this.renderLoading();
      await this.fetchRecommendations();
      this.render();
    }

    getHeaders() {
      const headers = {
        "Content-Type": "application/json",
      };
      if (this.config.apiKey) {
        headers["X-API-Key"] = this.config.apiKey;
      }
      if (this.config.tenantId) {
        headers["X-Tenant-ID"] = this.config.tenantId;
      }
      return headers;
    }

    async fetchRecommendations() {
      try {
        let url = `${this.config.apiUrl}${this.config.endpoint}`;
        const params = new URLSearchParams();
        if (this.config.userId) params.set("user_id", this.config.userId);
        if (this.config.k) params.set("k", this.config.k);

        const queryString = params.toString();
        if (queryString && !url.includes("?")) {
          url += `?${queryString}`;
        }

        const res = await fetch(url, { credentials: "same-origin", headers: this.getHeaders() });
        if (!res.ok) {
          throw new Error(`Failed to fetch recommendations: HTTP ${res.status}`);
        }
        const data = await res.json();
        this.items = data.recommendations || data.items || [];
      } catch (err) {
        console.error("SuisenShaWidget fetch error:", err);
        this.items = [];
      }
    }

    renderLoading() {
      this.container.innerHTML = `
        <div class="suisensha-carousel-container">
          <div class="suisensha-header">
            <h3 class="suisensha-title">${this.config.title}</h3>
          </div>
          <div style="padding: 24px; text-align: center; color: var(--ss-text-muted);">
            Loading personalized recommendations...
          </div>
        </div>
      `;
    }

    render() {
      if (!this.items || this.items.length === 0) {
        this.container.innerHTML = `
          <div class="suisensha-carousel-container">
            <div class="suisensha-header">
              <h3 class="suisensha-title">${this.config.title}</h3>
            </div>
            <div style="padding: 24px; text-align: center; color: var(--ss-text-muted);">
              No recommendations available at this time.
            </div>
          </div>
        `;
        return;
      }

      const cardsHtml = this.items
        .map((item) => {
          const scorePercent =
            item.score !== null && item.score !== undefined
              ? `${Math.round(item.score * 100)}% match`
              : "Recommended";

          return `
          <div class="suisensha-card" data-item-id="${item.item_id}">
            <div class="suisensha-card-top">
              <span class="suisensha-badge">${scorePercent}</span>
              <button class="suisensha-action-btn explain-btn" title="Why is this recommended? (DSA Transparency)" data-item-id="${item.item_id}" data-item-title="${item.title}">ℹ️</button>
            </div>
            <div class="suisensha-item-title" title="${item.title}">${item.title}</div>
            <div class="suisensha-item-genres">${item.genres || "General"}</div>
            <div class="suisensha-card-footer">
              <div class="suisensha-btn-group">
                <button class="suisensha-action-btn dislike dislike-btn" title="Not interested" data-item-id="${item.item_id}">✕</button>
              </div>
            </div>
          </div>
        `;
        })
        .join("");

      this.container.innerHTML = `
        <div class="suisensha-carousel-container">
          <div class="suisensha-header">
            <h3 class="suisensha-title">
              <span>✨ ${this.config.title}</span>
            </h3>
            <div class="suisensha-nav-controls">
              <button class="suisensha-nav-btn" id="ss-prev-btn" aria-label="Previous">‹</button>
              <button class="suisensha-nav-btn" id="ss-next-btn" aria-label="Next">›</button>
            </div>
          </div>
          <div class="suisensha-track" id="ss-track">
            ${cardsHtml}
          </div>
        </div>
      `;

      this.attachEvents();
    }

    attachEvents() {
      const track = this.container.querySelector("#ss-track");
      const prevBtn = this.container.querySelector("#ss-prev-btn");
      const nextBtn = this.container.querySelector("#ss-next-btn");

      if (track && prevBtn && nextBtn) {
        prevBtn.addEventListener("click", () => {
          track.scrollBy({ left: -300, behavior: "smooth" });
        });
        nextBtn.addEventListener("click", () => {
          track.scrollBy({ left: 300, behavior: "smooth" });
        });
      }

      // Card clicks for telemetry
      const cards = this.container.querySelectorAll(".suisensha-card");
      cards.forEach((card) => {
        card.addEventListener("click", (e) => {
          if (e.target.closest(".suisensha-action-btn")) return;
          const itemId = parseInt(card.getAttribute("data-item-id"), 10);
          this.recordEvent("click", itemId);
          if (typeof this.config.onItemClick === "function") {
            const item = this.items.find((it) => it.item_id === itemId);
            this.config.onItemClick(item);
          }
        });
      });

      // Dislike ("Not interested") button
      const dislikeBtns = this.container.querySelectorAll(".dislike-btn");
      dislikeBtns.forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const itemId = parseInt(btn.getAttribute("data-item-id"), 10);
          await this.recordEvent("not_interested", itemId);
          const card = btn.closest(".suisensha-card");
          if (card) {
            card.style.opacity = "0.3";
            card.style.pointerEvents = "none";
          }
          if (typeof this.config.onExclude === "function") {
            this.config.onExclude(itemId);
          }
        });
      });

      // Explainability modal
      const explainBtns = this.container.querySelectorAll(".explain-btn");
      explainBtns.forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const itemId = parseInt(btn.getAttribute("data-item-id"), 10);
          const itemTitle = btn.getAttribute("data-item-title");
          await this.showExplanationModal(itemId, itemTitle);
        });
      });
    }

    async recordEvent(eventType, itemId) {
      try {
        const payload = {
          events: [
            {
              item_id: itemId,
              event_type: eventType,
              user_id: this.config.userId || null,
              session_id: this.config.sessionId || null,
              tenant_id: this.config.tenantId || "default",
            },
          ],
        };

        await fetch(`${this.config.apiUrl}/v1/events`, {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify(payload),
        });
      } catch (err) {
        console.warn("SuisenShaWidget: Failed to log telemetry event:", err);
      }
    }

    async showExplanationModal(itemId, title) {
      let explanation =
        "Recommended based on your recent viewing sequence and similarity to titles you engaged with.";
      try {
        const res = await fetch(`${this.config.apiUrl}/v1/transparency`, {
          headers: this.getHeaders(),
        });
        if (res.ok) {
          const trans = await res.json();
          if (trans.main_parameters && trans.main_parameters.length > 0) {
            explanation = trans.main_parameters[0].description;
          }
        }
      } catch (err) {
        // Fallback explanation
      }

      const modalBackdrop = document.createElement("div");
      modalBackdrop.className = "suisensha-modal-backdrop";
      modalBackdrop.innerHTML = `
        <div class="suisensha-modal">
          <div class="suisensha-modal-header">
            <h4 class="suisensha-modal-title">Why this recommendation?</h4>
            <button class="suisensha-modal-close">&times;</button>
          </div>
          <div class="suisensha-modal-body">
            <p><strong>Title:</strong> ${title}</p>
            <p><strong>Rationale:</strong> ${explanation}</p>
            <p style="font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
              Digital Services Act (EU 2022/2065) Art. 27 Recommender Transparency Notice.
            </p>
          </div>
        </div>
      `;

      modalBackdrop.querySelector(".suisensha-modal-close").addEventListener("click", () => {
        modalBackdrop.remove();
      });
      modalBackdrop.addEventListener("click", (e) => {
        if (e.target === modalBackdrop) modalBackdrop.remove();
      });

      document.body.appendChild(modalBackdrop);
    }
  }

  global.SuisenShaWidget = SuisenShaWidget;
})(typeof window !== "undefined" ? window : this);
