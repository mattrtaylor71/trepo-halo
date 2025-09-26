// Simplified shopping.js for local development
// Removes all backend API dependencies and uses mock data

/**********************************************
 * MOCK DATA
 **********************************************/
let mockShoppingItems = [
    {
        id: "1",
        name: "Organic Bananas",
        brand: "Whole Foods Market",
        category: "Fruits",
        image: "img/image-6.png",
        inInventory: false,
        inShoppingList: true
    },
    {
        id: "2",
        name: "Avocados",
        brand: "Organic Valley",
        category: "Fruits",
        image: "img/image-7.png",
        inInventory: false,
        inShoppingList: true
    },
    {
        id: "3",
        name: "Greek Yogurt",
        brand: "Chobani",
        category: "Dairy",
        image: "img/image-8.png",
        inInventory: true,
        inShoppingList: false
    },
    {
        id: "4",
        name: "Quinoa",
        brand: "Ancient Harvest",
        category: "Grains",
        image: "img/image-9.png",
        inInventory: false,
        inShoppingList: true
    },
    {
        id: "5",
        name: "Spinach",
        brand: "Organic Valley",
        category: "Vegetables",
        image: "img/image-11.png",
        inInventory: true,
        inShoppingList: false
    }
];

let userShoppingList = {};

/**********************************************
 * OVERLAY FUNCTIONS
 **********************************************/
function showOverlay(overlayId) {
    const overlay = document.getElementById(overlayId);
    if (overlay) {
        overlay.style.display = "flex";
    }
}

function hideOverlay(overlayId) {
    const overlay = document.getElementById(overlayId);
    if (overlay) {
        overlay.style.display = "none";
    }
}

/**********************************************
 * MOCK SEARCH FUNCTION
 **********************************************/
async function fetchWholeFoodsItems(query) {
    console.log("Mock search for:", query);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Return mock search results
    const mockResults = [
        {
            name: `${query} - Organic`,
            brand: "Whole Foods Market",
            imageThumbnail: "img/image-6.png"
        },
        {
            name: `${query} - Premium`,
            brand: "Organic Valley",
            imageThumbnail: "img/image-7.png"
        },
        {
            name: `${query} - Natural`,
            brand: "Nature's Best",
            imageThumbnail: "img/image-8.png"
        }
    ];
    
    return mockResults;
}

/**********************************************
 * RENDER SEARCH RESULTS IN OVERLAY
 **********************************************/
function renderSearchResults(items) {
    const resultsContainer = document.createElement("div");
    resultsContainer.className = "results-container";

    if (items.length === 0) {
        resultsContainer.innerHTML = "<p>No items found. Try a different search term.</p>";
    } else {
        items.forEach((item) => {
            const itemElement = document.createElement("div");
            itemElement.className = "result-item";

            itemElement.innerHTML = `
                <div class="result-item-row">
                    <img 
                        src="${item.imageThumbnail || ''}" 
                        alt="${item.name}" 
                        class="result-item-image" 
                    />
                    <div class="result-item-info">
                        <p class="result-item-name">${item.name}</p>
                        <p class="result-item-brand">${item.brand || 'Unknown Brand'}</p>
                    </div>
                </div>
            `;

            itemElement.addEventListener("click", () => {
                addItemToShoppingList(item);
                hideOverlay("search-overlay");
            });

            resultsContainer.appendChild(itemElement);
        });
    }

    const existingResults = document.querySelector(".results-container");
    if (existingResults) {
        existingResults.remove();
    }

    const overlayContent = document.querySelector(".search-overlay-content");
    if (overlayContent) {
        overlayContent.appendChild(resultsContainer);
    }
}

/**********************************************
 * ADD ITEM TO SHOPPING LIST
 **********************************************/
function addItemToShoppingList(item) {
    const newItem = {
        id: Date.now().toString(),
        name: item.name,
        brand: item.brand,
        category: "Other",
        image: item.imageThumbnail,
        inInventory: false,
        inShoppingList: true
    };
    
    mockShoppingItems.push(newItem);
    renderShoppingList();
    console.log("Added item to shopping list:", newItem);
}

/**********************************************
 * FETCH SHOPPING ITEMS (MOCK)
 **********************************************/
async function fetchShoppingItems() {
    console.log("Fetching mock shopping items...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return mockShoppingItems;
}

/**********************************************
 * GROUP ITEMS BY CATEGORY
 **********************************************/
function groupItemsByCategory(items) {
    const grouped = {};
    
    items.forEach(item => {
        const category = item.category || "Other";
        if (!grouped[category]) {
            grouped[category] = [];
        }
        grouped[category].push(item);
    });
    
    return grouped;
}

/**********************************************
 * RENDER ITEMS BY CATEGORY
 **********************************************/
function renderItemsByCategoryWithoutHeader(groupedItems, container) {
    container.innerHTML = "";
    
    Object.keys(groupedItems).forEach(category => {
        const categoryItems = groupedItems[category];
        
        categoryItems.forEach(item => {
            const itemElement = createItemElement(item);
            container.appendChild(itemElement);
        });
    });
}

/**********************************************
 * CREATE ITEM ELEMENT
 **********************************************/
function createItemElement(item) {
    const itemDiv = document.createElement("div");
    itemDiv.className = "shopping-item";
    itemDiv.dataset.itemId = item.id;
    
    const inventoryClass = item.inInventory ? "in-inventory" : "";
    const shoppingClass = item.inShoppingList ? "in-shopping-list" : "";
    
    itemDiv.innerHTML = `
        <div class="item-content ${inventoryClass} ${shoppingClass}">
            <img src="${item.image}" alt="${item.name}" class="item-image" />
            <div class="item-details">
                <h3 class="item-name">${item.name}</h3>
                <p class="item-brand">${item.brand}</p>
                <p class="item-category">${item.category}</p>
            </div>
            <div class="item-actions">
                <button class="inventory-btn ${item.inInventory ? 'active' : ''}" 
                        onclick="toggleInventory('${item.id}')">
                    ${item.inInventory ? '✓' : '○'}
                </button>
                <button class="shopping-btn ${item.inShoppingList ? 'active' : ''}" 
                        onclick="toggleShoppingList('${item.id}')">
                    ${item.inShoppingList ? '✓' : '○'}
                </button>
            </div>
        </div>
    `;
    
    return itemDiv;
}

/**********************************************
 * TOGGLE INVENTORY
 **********************************************/
async function toggleInventory(itemId) {
    const item = mockShoppingItems.find(i => i.id === itemId);
    if (item) {
        item.inInventory = !item.inInventory;
        renderShoppingList();
        console.log(`Toggled inventory for ${item.name}: ${item.inInventory}`);
    }
}

/**********************************************
 * TOGGLE SHOPPING LIST
 **********************************************/
async function toggleShoppingList(itemId) {
    const item = mockShoppingItems.find(i => i.id === itemId);
    if (item) {
        item.inShoppingList = !item.inShoppingList;
        renderShoppingList();
        console.log(`Toggled shopping list for ${item.name}: ${item.inShoppingList}`);
    }
}

/**********************************************
 * RENDER SHOPPING LIST
 **********************************************/
async function renderShoppingList() {
    const container = document.getElementById("shopping-list-container");
    if (!container) return;
    
    const items = await fetchShoppingItems();
    const groupedItems = groupItemsByCategory(items);
    
    renderItemsByCategoryWithoutHeader(groupedItems, container);
}

/**********************************************
 * SEARCH FUNCTIONALITY
 **********************************************/
async function handleSearch() {
    const searchInput = document.getElementById("search-input");
    const query = searchInput.value.trim();
    
    if (!query) {
        alert("Please enter a search term");
        return;
    }
    
    showOverlay("search-overlay");
    
    try {
        const results = await fetchWholeFoodsItems(query);
        renderSearchResults(results);
    } catch (error) {
        console.error("Search error:", error);
        alert("Search failed. Please try again.");
    }
}

/**********************************************
 * CREATE INSTACART SHOPPING LIST (MOCK)
 **********************************************/
async function createInstacartShoppingList() {
    console.log("Creating mock Instacart shopping list...");
    
    const shoppingItems = mockShoppingItems.filter(item => item.inShoppingList);
    
    if (shoppingItems.length === 0) {
        alert("No items in your shopping list!");
        return;
    }
    
    // Mock Instacart list creation
    const listUrl = "https://www.instacart.com/store/whole-foods-market/storefront";
    
    alert(`Mock: Redirecting to Instacart with ${shoppingItems.length} items in your list!`);
    console.log("Items for Instacart:", shoppingItems.map(item => item.name));
    
    // In a real implementation, this would open the Instacart URL
    // window.open(listUrl, '_blank');
}

/**********************************************
 * HIGHLIGHT ALL BUTTON
 **********************************************/
function highlightAllButton() {
    const items = document.querySelectorAll(".shopping-item");
    items.forEach(item => {
        item.style.backgroundColor = "#e8f5e8";
        setTimeout(() => {
            item.style.backgroundColor = "";
        }, 2000);
    });
}

/**********************************************
 * FETCH AI INSIGHTS (MOCK)
 **********************************************/
async function fetchAIInsights() {
    console.log("Fetching mock AI insights...");
    
    const insightsContainer = document.getElementById("ai-insights");
    if (!insightsContainer) return;
    
    // Mock AI insights
    const mockInsights = `
        <div class="ai-insight">
            <h3>🤖 AI Shopping Assistant</h3>
            <p>Based on your shopping patterns, here are some recommendations:</p>
            <ul>
                <li>Consider adding more leafy greens to your cart</li>
                <li>You're doing great with organic choices!</li>
                <li>Try exploring the bulk section for better value</li>
                <li>Don't forget to check for seasonal produce</li>
            </ul>
        </div>
    `;
    
    insightsContainer.innerHTML = mockInsights;
}

/**********************************************
 * INITIALIZE PAGE
 **********************************************/
document.addEventListener("DOMContentLoaded", async () => {
    console.log("Initializing shopping page...");
    
    // Set up search functionality
    const searchBtn = document.getElementById("search-btn");
    if (searchBtn) {
        searchBtn.addEventListener("click", handleSearch);
    }
    
    // Set up search input enter key
    const searchInput = document.getElementById("search-input");
    if (searchInput) {
        searchInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                handleSearch();
            }
        });
    }
    
    // Set up Instacart button
    const instacartBtn = document.getElementById("instacart-btn");
    if (instacartBtn) {
        instacartBtn.addEventListener("click", createInstacartShoppingList);
    }
    
    // Set up highlight all button
    const highlightBtn = document.getElementById("highlight-all-btn");
    if (highlightBtn) {
        highlightBtn.addEventListener("click", highlightAllButton);
    }
    
    // Close overlay when clicking outside
    const searchOverlay = document.getElementById("search-overlay");
    if (searchOverlay) {
        searchOverlay.addEventListener("click", (e) => {
            if (e.target === searchOverlay) {
                hideOverlay("search-overlay");
            }
        });
    }
    
    // Initial render
    await renderShoppingList();
    await fetchAIInsights();
});




