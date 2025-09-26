// Simplified points.js for local development
// Removes all backend API dependencies and uses mock data

/**********************************************
 * MOCK DATA
 **********************************************/
let mockPointsData = {
    currentPoints: 1250,
    totalEarned: 1800,
    totalSpent: 550,
    history: [
        {
            id: "1",
            action: "earned",
            amount: 50,
            description: "Scanned Organic Bananas",
            timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
            category: "Fruits"
        },
        {
            id: "2",
            action: "earned",
            amount: 75,
            description: "Scanned Greek Yogurt",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
            category: "Dairy"
        },
        {
            id: "3",
            action: "spent",
            amount: -100,
            description: "Redeemed: $10 Amazon Gift Card",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
            category: "Reward"
        },
        {
            id: "4",
            action: "earned",
            amount: 90,
            description: "Scanned Quinoa",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2), // 2 days ago
            category: "Grains"
        },
        {
            id: "5",
            action: "earned",
            amount: 100,
            description: "Scanned Spinach",
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3), // 3 days ago
            category: "Vegetables"
        }
    ],
    rewards: [
        {
            id: "1",
            name: "$5 Amazon Gift Card",
            cost: 500,
            available: true,
            image: "img/amazon.png"
        },
        {
            id: "2",
            name: "$10 Amazon Gift Card",
            cost: 1000,
            available: true,
            image: "img/amazon.png"
        },
        {
            id: "3",
            name: "$25 Amazon Gift Card",
            cost: 2500,
            available: false,
            image: "img/amazon.png"
        },
        {
            id: "4",
            name: "Free Shipping",
            cost: 300,
            available: true,
            image: "img/amazon.png"
        }
    ]
};

/**********************************************
 * FETCH POINTS DATA (MOCK)
 **********************************************/
async function fetchPointsData() {
    console.log("Fetching mock points data...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return mockPointsData;
}

/**********************************************
 * RENDER POINTS OVERVIEW
 **********************************************/
async function renderPointsOverview() {
    const container = document.getElementById("points-overview-container");
    if (!container) return;
    
    const data = await fetchPointsData();
    
    container.innerHTML = `
        <div class="points-summary">
            <h2>Your Points</h2>
            <div class="points-display">
                <div class="current-points">
                    <span class="points-number">${data.currentPoints}</span>
                    <span class="points-label">Current Points</span>
                </div>
                <div class="points-stats">
                    <div class="stat-item">
                        <span class="stat-number">${data.totalEarned}</span>
                        <span class="stat-label">Total Earned</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${data.totalSpent}</span>
                        <span class="stat-label">Total Spent</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER POINTS HISTORY
 **********************************************/
async function renderPointsHistory() {
    const container = document.getElementById("points-history-container");
    if (!container) return;
    
    const data = await fetchPointsData();
    
    container.innerHTML = `
        <div class="history-section">
            <h3>Points History</h3>
            <div class="history-list">
                ${data.history.map(entry => `
                    <div class="history-item ${entry.action}">
                        <div class="history-content">
                            <div class="history-info">
                                <h4 class="history-description">${entry.description}</h4>
                                <p class="history-category">${entry.category}</p>
                                <p class="history-time">${formatTimeAgo(entry.timestamp)}</p>
                            </div>
                            <div class="history-amount">
                                <span class="amount ${entry.action}">
                                    ${entry.action === 'earned' ? '+' : ''}${entry.amount}
                                </span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * RENDER REWARDS
 **********************************************/
async function renderRewards() {
    const container = document.getElementById("rewards-container");
    if (!container) return;
    
    const data = await fetchPointsData();
    
    container.innerHTML = `
        <div class="rewards-section">
            <h3>Available Rewards</h3>
            <div class="rewards-grid">
                ${data.rewards.map(reward => `
                    <div class="reward-card ${reward.available ? 'available' : 'unavailable'}">
                        <div class="reward-image">
                            <img src="${reward.image}" alt="${reward.name}">
                        </div>
                        <div class="reward-info">
                            <h4 class="reward-name">${reward.name}</h4>
                            <p class="reward-cost">${reward.cost} points</p>
                        </div>
                        <button class="redeem-btn ${reward.available ? '' : 'disabled'}" 
                                onclick="redeemReward('${reward.id}')"
                                ${!reward.available ? 'disabled' : ''}>
                            ${reward.available ? 'Redeem' : 'Not Enough Points'}
                        </button>
                    </div>
                `).join('')}
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
 * REDEEM REWARD (MOCK)
 **********************************************/
async function redeemReward(rewardId) {
    console.log(`Redeeming reward ${rewardId}...`);
    
    const reward = mockPointsData.rewards.find(r => r.id === rewardId);
    if (!reward || !reward.available) {
        alert("This reward is not available for redemption.");
        return;
    }
    
    if (mockPointsData.currentPoints < reward.cost) {
        alert("Not enough points to redeem this reward.");
        return;
    }
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Update mock data
    mockPointsData.currentPoints -= reward.cost;
    mockPointsData.totalSpent += reward.cost;
    
    // Add to history
    mockPointsData.history.unshift({
        id: Date.now().toString(),
        action: "spent",
        amount: -reward.cost,
        description: `Redeemed: ${reward.name}`,
        timestamp: new Date(),
        category: "Reward"
    });
    
    // Re-render sections
    await renderPointsOverview();
    await renderPointsHistory();
    await renderRewards();
    
    alert(`Successfully redeemed ${reward.name}! Check your email for details.`);
    console.log(`Reward ${rewardId} redeemed successfully`);
}

/**********************************************
 * INITIALIZE PAGE
 **********************************************/
document.addEventListener("DOMContentLoaded", async () => {
    console.log("Initializing points page...");
    
    // Render all sections
    await renderPointsOverview();
    await renderPointsHistory();
    await renderRewards();
    
    console.log("Points page initialized successfully");
});
