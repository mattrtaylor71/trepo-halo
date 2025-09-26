// Simplified beeps.js for local development
// Removes all backend API dependencies and uses mock data

/**********************************************
 * MOCK DATA
 **********************************************/
let mockBeepsData = {
    recentBeeps: [
        {
            id: "1",
            item: "Organic Bananas",
            timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
            score: 90,
            category: "Fruits"
        },
        {
            id: "2",
            item: "Greek Yogurt",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
            score: 85,
            category: "Dairy"
        },
        {
            id: "3",
            item: "Quinoa",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4), // 4 hours ago
            score: 95,
            category: "Grains"
        },
        {
            id: "4",
            item: "Spinach",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6), // 6 hours ago
            score: 100,
            category: "Vegetables"
        },
        {
            id: "5",
            item: "Avocados",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 8), // 8 hours ago
            score: 88,
            category: "Fruits"
        }
    ],
    stats: {
        totalBeeps: 156,
        thisWeek: 23,
        averageScore: 87,
        topCategory: "Fruits"
    }
};

/**********************************************
 * FETCH BEEPS DATA (MOCK)
 **********************************************/
async function fetchBeepsData() {
    console.log("Fetching mock beeps data...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return mockBeepsData;
}

/**********************************************
 * RENDER BEEPS LIST
 **********************************************/
async function renderBeepsList() {
    const container = document.getElementById("beeps-list-container");
    if (!container) return;
    
    const data = await fetchBeepsData();
    
    container.innerHTML = `
        <div class="beeps-header">
            <h2>Recent Scans</h2>
            <p>Your latest product scans and their health scores</p>
        </div>
        <div class="beeps-list">
            ${data.recentBeeps.map(beep => `
                <div class="beep-item">
                    <div class="beep-content">
                        <div class="beep-info">
                            <h3 class="beep-name">${beep.item}</h3>
                            <p class="beep-category">${beep.category}</p>
                            <p class="beep-time">${formatTimeAgo(beep.timestamp)}</p>
                        </div>
                        <div class="beep-score">
                            <div class="score-circle">
                                <span class="score-number">${beep.score}</span>
                            </div>
                            <span class="score-label">/100</span>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**********************************************
 * RENDER STATS
 **********************************************/
async function renderStats() {
    const container = document.getElementById("stats-container");
    if (!container) return;
    
    const data = await fetchBeepsData();
    
    container.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">${data.stats.totalBeeps}</div>
                <div class="stat-label">Total Scans</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${data.stats.thisWeek}</div>
                <div class="stat-label">This Week</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${data.stats.averageScore}</div>
                <div class="stat-label">Avg Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${data.stats.topCategory}</div>
                <div class="stat-label">Top Category</div>
            </div>
        </div>
    `;
}

/**********************************************
 * FORMAT TIME AGO
 **********************************************/
function formatTimeAgo(timestamp) {
    const now = new Date();
    const diff = now - timestamp;
    
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (minutes < 60) {
        return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    } else if (hours < 24) {
        return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    } else {
        return `${days} day${days !== 1 ? 's' : ''} ago`;
    }
}

/**********************************************
 * SEND REWARD EMAIL (MOCK)
 **********************************************/
async function sendRewardEmail() {
    console.log("Sending mock reward email...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Show success message
    const messageContainer = document.getElementById("email-message");
    if (messageContainer) {
        messageContainer.innerHTML = `
            <div class="success-message">
                <h3>🎉 Email Sent Successfully!</h3>
                <p>Your reward summary has been sent to your email address.</p>
                <p>Check your inbox for detailed insights about your consumption patterns.</p>
            </div>
        `;
    }
    
    console.log("Mock reward email sent successfully");
}

/**********************************************
 * INITIALIZE PAGE
 **********************************************/
document.addEventListener("DOMContentLoaded", async () => {
    console.log("Initializing beeps page...");
    
    // Render all sections
    await renderBeepsList();
    await renderStats();
    
    // Set up event listeners
    const emailBtn = document.getElementById("send-email-btn");
    if (emailBtn) {
        emailBtn.addEventListener("click", sendRewardEmail);
    }
    
    console.log("Beeps page initialized successfully");
});
