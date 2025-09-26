// Simplified week.js for local development
// Removes all backend API dependencies and uses mock data

/**********************************************
 * MOCK DATA
 **********************************************/
let mockWeekData = {
    currentWeek: {
        score: 85,
        items: [
            { name: "Organic Bananas", category: "Fruits", score: 90 },
            { name: "Greek Yogurt", category: "Dairy", score: 85 },
            { name: "Quinoa", category: "Grains", score: 95 },
            { name: "Spinach", category: "Vegetables", score: 100 },
            { name: "Avocados", category: "Fruits", score: 88 }
        ]
    },
    previousWeeks: [
        { week: "Week 1", score: 75, items: 18 },
        { week: "Week 2", score: 82, items: 22 },
        { week: "Week 3", score: 78, items: 20 },
        { week: "Week 4", score: 85, items: 25 },
        { week: "Week 5", score: 88, items: 23 }
    ],
    alternates: [
        { name: "Regular Bananas", better: "Organic Bananas", reason: "No pesticides" },
        { name: "Regular Yogurt", better: "Greek Yogurt", reason: "Higher protein content" },
        { name: "White Rice", better: "Quinoa", reason: "More nutrients and fiber" }
    ]
};

/**********************************************
 * FETCH WEEK DATA (MOCK)
 **********************************************/
async function fetchWeekData() {
    console.log("Fetching mock week data...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return mockWeekData;
}

/**********************************************
 * FETCH ALTERNATES (MOCK)
 **********************************************/
async function fetchAlternates() {
    console.log("Fetching mock alternates...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    return mockWeekData.alternates;
}

/**********************************************
 * FETCH SHOPPING ITEMS (MOCK)
 **********************************************/
async function fetchShoppingItems() {
    console.log("Fetching mock shopping items...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 250));
    
    return mockWeekData.currentWeek.items;
}

/**********************************************
 * RENDER WEEK OVERVIEW
 **********************************************/
async function renderWeekOverview() {
    const container = document.getElementById("week-overview-container");
    if (!container) return;
    
    const data = await fetchWeekData();
    
    container.innerHTML = `
        <div class="week-summary">
            <h2>This Week's Summary</h2>
            <div class="score-display">
                <div class="score-circle">
                    <span class="score-number">${data.currentWeek.score}</span>
                    <span class="score-label">/100</span>
                </div>
                <p class="score-description">Your consumption score this week</p>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-number">${data.currentWeek.items.length}</span>
                    <span class="stat-label">Items Scanned</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${Math.round(data.currentWeek.score / 10)}</span>
                    <span class="stat-label">Health Rating</span>
                </div>
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER WEEKLY CHART
 **********************************************/
async function renderWeeklyChart() {
    const container = document.getElementById("weekly-chart-container");
    if (!container) return;
    
    const data = await fetchWeekData();
    
    container.innerHTML = `
        <div class="chart-container">
            <h3>Weekly Progress</h3>
            <div class="chart-bars">
                ${data.previousWeeks.map((week, index) => `
                    <div class="chart-bar-container">
                        <div class="chart-bar" style="height: ${week.score}%">
                            <span class="bar-score">${week.score}</span>
                        </div>
                        <span class="bar-label">${week.week}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER ITEMS LIST
 **********************************************/
async function renderItemsList() {
    const container = document.getElementById("items-list-container");
    if (!container) return;
    
    const data = await fetchWeekData();
    
    container.innerHTML = `
        <div class="items-section">
            <h3>This Week's Items</h3>
            <div class="items-grid">
                ${data.currentWeek.items.map(item => `
                    <div class="item-card">
                        <div class="item-header">
                            <h4>${item.name}</h4>
                            <span class="item-category">${item.category}</span>
                        </div>
                        <div class="item-score">
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${item.score}%"></div>
                            </div>
                            <span class="score-text">${item.score}/100</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER ALTERNATES
 **********************************************/
async function renderAlternates() {
    const container = document.getElementById("alternates-container");
    if (!container) return;
    
    const alternates = await fetchAlternates();
    
    container.innerHTML = `
        <div class="alternates-section">
            <h3>Better Alternatives</h3>
            <div class="alternates-list">
                ${alternates.map(alt => `
                    <div class="alternate-item">
                        <div class="alternate-content">
                            <div class="alternate-before">
                                <span class="label">Instead of:</span>
                                <span class="item-name">${alt.name}</span>
                            </div>
                            <div class="alternate-arrow">→</div>
                            <div class="alternate-after">
                                <span class="label">Try:</span>
                                <span class="item-name better">${alt.better}</span>
                            </div>
                        </div>
                        <div class="alternate-reason">
                            <span class="reason-label">Why:</span>
                            <span class="reason-text">${alt.reason}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER SHOPPING SUGGESTIONS
 **********************************************/
async function renderShoppingSuggestions() {
    const container = document.getElementById("shopping-suggestions-container");
    if (!container) return;
    
    const items = await fetchShoppingItems();
    const suggestions = items.filter(item => item.score < 80).slice(0, 5);
    
    container.innerHTML = `
        <div class="suggestions-section">
            <h3>Shopping Suggestions</h3>
            <p>Based on your consumption patterns, consider these items for next week:</p>
            <div class="suggestions-list">
                ${suggestions.map(item => `
                    <div class="suggestion-item">
                        <div class="suggestion-info">
                            <span class="suggestion-name">${item.name}</span>
                            <span class="suggestion-category">${item.category}</span>
                        </div>
                        <div class="suggestion-score">
                            <span class="current-score">${item.score}/100</span>
                            <span class="improvement">+${100 - item.score} points potential</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * TOGGLE OVERLAY
 **********************************************/
function toggleOverlay(overlayId, show = true) {
    const overlay = document.getElementById(overlayId);
    if (!overlay) return;
    
    if (show) {
        overlay.style.display = "flex";
    } else {
        overlay.style.display = "none";
    }
}

/**********************************************
 * SHOW DETAILED VIEW
 **********************************************/
async function showDetailedView(type) {
    const overlay = document.getElementById("detailed-overlay");
    const content = document.getElementById("detailed-content");
    
    if (!overlay || !content) return;
    
    let contentHTML = "";
    
    switch (type) {
        case "items":
            const data = await fetchWeekData();
            contentHTML = `
                <div class="detailed-items">
                    <h2>Detailed Item Analysis</h2>
                    <div class="items-detail-list">
                        ${data.currentWeek.items.map(item => `
                            <div class="detail-item">
                                <div class="item-info">
                                    <h3>${item.name}</h3>
                                    <p class="category">${item.category}</p>
                                </div>
                                <div class="item-analysis">
                                    <div class="score-breakdown">
                                        <span class="score">${item.score}/100</span>
                                        <div class="score-bar">
                                            <div class="score-fill" style="width: ${item.score}%"></div>
                                        </div>
                                    </div>
                                    <div class="analysis-text">
                                        <p>This item contributes to your overall health score. 
                                        ${item.score >= 80 ? 'Great choice!' : 'Consider alternatives for better nutrition.'}</p>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            break;
            
        case "alternates":
            const alternates = await fetchAlternates();
            contentHTML = `
                <div class="detailed-alternates">
                    <h2>Alternative Recommendations</h2>
                    <div class="alternates-detail-list">
                        ${alternates.map(alt => `
                            <div class="detail-alternate">
                                <div class="alternate-comparison">
                                    <div class="current-choice">
                                        <h3>Current Choice</h3>
                                        <p>${alt.name}</p>
                                    </div>
                                    <div class="arrow">→</div>
                                    <div class="better-choice">
                                        <h3>Better Alternative</h3>
                                        <p>${alt.better}</p>
                                    </div>
                                </div>
                                <div class="reasoning">
                                    <h4>Why this is better:</h4>
                                    <p>${alt.reason}</p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            break;
            
        default:
            contentHTML = "<p>No detailed view available for this section.</p>";
    }
    
    content.innerHTML = contentHTML;
    toggleOverlay("detailed-overlay", true);
}

/**********************************************
 * INITIALIZE PAGE
 **********************************************/
document.addEventListener("DOMContentLoaded", async () => {
    console.log("Initializing week page...");
    
    // Render all sections
    await renderWeekOverview();
    await renderWeeklyChart();
    await renderItemsList();
    await renderAlternates();
    await renderShoppingSuggestions();
    
    // Set up event listeners
    const itemsBtn = document.getElementById("view-items-btn");
    const alternatesBtn = document.getElementById("view-alternates-btn");
    const closeBtn = document.getElementById("close-overlay-btn");
    
    if (itemsBtn) {
        itemsBtn.addEventListener("click", () => showDetailedView("items"));
    }
    
    if (alternatesBtn) {
        alternatesBtn.addEventListener("click", () => showDetailedView("alternates"));
    }
    
    if (closeBtn) {
        closeBtn.addEventListener("click", () => toggleOverlay("detailed-overlay", false));
    }
    
    // Close overlay when clicking outside
    const overlay = document.getElementById("detailed-overlay");
    if (overlay) {
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                toggleOverlay("detailed-overlay", false);
            }
        });
    }
    
    console.log("Week page initialized successfully");
});

