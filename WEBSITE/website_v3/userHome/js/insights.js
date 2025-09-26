// Simplified insights.js for local development
// Hyper-personalized grocery shopping page

// Sample data for the grocery shopping page
const groceryData = {
  outOfItems: [
    { name: "Organic Bananas", category: "Fruits", price: "$2.99", image: "🍌", urgency: "high" },
    { name: "Greek Yogurt", category: "Dairy", price: "$4.49", image: "🥛", urgency: "high" },
    { name: "Whole Grain Bread", category: "Bakery", price: "$3.99", image: "🍞", urgency: "medium" },
    { name: "Spinach", category: "Vegetables", price: "$2.49", image: "🥬", urgency: "high" },
    { name: "Chicken Breast", category: "Meat", price: "$8.99", image: "🍗", urgency: "medium" },
    { name: "Quinoa", category: "Grains", price: "$5.99", image: "🌾", urgency: "low" }
  ],
  
  mealPlan: [
    { day: "Monday", meal: "Breakfast", name: "Greek Yogurt Parfait", ingredients: ["Greek yogurt", "Berries", "Granola"], image: "🥣" },
    { day: "Monday", meal: "Lunch", name: "Quinoa Buddha Bowl", ingredients: ["Quinoa", "Chickpeas", "Avocado"], image: "🥗" },
    { day: "Monday", meal: "Dinner", name: "Grilled Chicken Salad", ingredients: ["Chicken breast", "Mixed greens", "Cherry tomatoes"], image: "🥗" },
    { day: "Tuesday", meal: "Breakfast", name: "Smoothie Bowl", ingredients: ["Banana", "Spinach", "Almond milk"], image: "🥤" },
    { day: "Tuesday", meal: "Lunch", name: "Mediterranean Wrap", ingredients: ["Whole grain wrap", "Hummus", "Cucumber"], image: "🌯" },
    { day: "Tuesday", meal: "Dinner", name: "Salmon with Roasted Vegetables", ingredients: ["Salmon", "Broccoli", "Sweet potato"], image: "🐟" }
  ],
  
  recipeRecommendations: [
    { name: "Quinoa Buddha Bowl", time: "25 min", difficulty: "Easy", rating: 4.8, image: "🥗", tags: ["Vegetarian", "High Protein"] },
    { name: "Mediterranean Chicken", time: "35 min", difficulty: "Medium", rating: 4.6, image: "🍗", tags: ["Gluten-Free", "Low Carb"] },
    { name: "Berry Smoothie Bowl", time: "10 min", difficulty: "Easy", rating: 4.9, image: "🥤", tags: ["Vegan", "Quick"] },
    { name: "Roasted Vegetable Pasta", time: "40 min", difficulty: "Medium", rating: 4.7, image: "🍝", tags: ["Vegetarian", "Comfort Food"] }
  ],
  
  smartRecommendations: [
    { name: "Organic Avocados", reason: "You love healthy fats", price: "$3.99", image: "🥑", category: "Fruits" },
    { name: "Almond Butter", reason: "Perfect for your smoothies", price: "$6.99", image: "🥜", category: "Pantry" },
    { name: "Chia Seeds", reason: "Great for your yogurt bowls", price: "$4.99", image: "🌱", category: "Pantry" },
    { name: "Sweet Potatoes", reason: "You buy these weekly", price: "$2.99", image: "🍠", category: "Vegetables" },
    { name: "Coconut Water", reason: "Perfect post-workout drink", price: "$3.49", image: "🥥", category: "Beverages" },
    { name: "Dark Chocolate", reason: "Your healthy treat choice", price: "$4.99", image: "🍫", category: "Snacks" }
  ],
  
  shoppingList: {
    "Fruits & Vegetables": [
      { name: "Organic Bananas", quantity: "1 bunch", price: "$2.99", checked: false },
      { name: "Spinach", quantity: "1 bag", price: "$2.49", checked: false },
      { name: "Avocados", quantity: "3 pieces", price: "$3.99", checked: false },
      { name: "Sweet Potatoes", quantity: "2 lbs", price: "$2.99", checked: false }
    ],
    "Dairy & Eggs": [
      { name: "Greek Yogurt", quantity: "2 containers", price: "$4.49", checked: false },
      { name: "Organic Eggs", quantity: "1 dozen", price: "$5.99", checked: false }
    ],
    "Meat & Fish": [
      { name: "Chicken Breast", quantity: "1 lb", price: "$8.99", checked: false },
      { name: "Salmon Fillets", quantity: "2 pieces", price: "$12.99", checked: false }
    ],
    "Pantry": [
      { name: "Whole Grain Bread", quantity: "1 loaf", price: "$3.99", checked: false },
      { name: "Quinoa", quantity: "1 bag", price: "$5.99", checked: false },
      { name: "Almond Butter", quantity: "1 jar", price: "$6.99", checked: false }
    ]
  }
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
  populateOutOfItems();
  populateMealPlan();
  populateRecipeRecommendations();
  populateSmartRecommendations();
  populateShoppingList();
});

// Populate "Items You're Out Of" section
function populateOutOfItems() {
  const container = document.getElementById('out-of-items');
  if (!container) return;
  
  container.innerHTML = groceryData.outOfItems.map(item => `
    <div class="item-card ${item.urgency}">
      <div class="item-emoji">${item.image}</div>
      <div class="item-info">
        <h3 class="item-name">${item.name}</h3>
        <p class="item-category">${item.category}</p>
        <p class="item-price">${item.price}</p>
      </div>
      <button class="add-to-cart-btn" onclick="addToCart('${item.name}')">Add</button>
    </div>
  `).join('');
}

// Populate meal plan section
function populateMealPlan() {
  const container = document.getElementById('meal-plan');
  if (!container) return;
  
  const groupedMeals = {};
  groceryData.mealPlan.forEach(meal => {
    if (!groupedMeals[meal.day]) {
      groupedMeals[meal.day] = [];
    }
    groupedMeals[meal.day].push(meal);
  });
  
  container.innerHTML = Object.entries(groupedMeals).map(([day, meals]) => `
    <div class="meal-day-card">
      <h3 class="day-title">${day}</h3>
      ${meals.map(meal => `
        <div class="meal-item">
          <div class="meal-emoji">${meal.image}</div>
          <div class="meal-details">
            <h4 class="meal-name">${meal.name}</h4>
            <p class="meal-type">${meal.meal}</p>
            <p class="meal-ingredients">${meal.ingredients.join(', ')}</p>
          </div>
        </div>
      `).join('')}
    </div>
  `).join('');
}

// Populate recipe recommendations
function populateRecipeRecommendations() {
  const container = document.getElementById('recipe-recommendations');
  if (!container) return;
  
  container.innerHTML = groceryData.recipeRecommendations.map(recipe => `
    <div class="recipe-card">
      <div class="recipe-emoji">${recipe.image}</div>
      <div class="recipe-info">
        <h3 class="recipe-name">${recipe.name}</h3>
        <div class="recipe-meta">
          <span class="recipe-time">⏱️ ${recipe.time}</span>
          <span class="recipe-difficulty">📊 ${recipe.difficulty}</span>
          <span class="recipe-rating">⭐ ${recipe.rating}</span>
        </div>
        <div class="recipe-tags">
          ${recipe.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
        </div>
      </div>
      <button class="view-recipe-btn" onclick="viewRecipe('${recipe.name}')">View Recipe</button>
    </div>
  `).join('');
}

// Populate smart recommendations
function populateSmartRecommendations() {
  const container = document.getElementById('smart-recommendations');
  if (!container) return;
  
  container.innerHTML = groceryData.smartRecommendations.map(item => `
    <div class="recommendation-card">
      <div class="recommendation-emoji">${item.image}</div>
      <div class="recommendation-info">
        <h3 class="recommendation-name">${item.name}</h3>
        <p class="recommendation-reason">${item.reason}</p>
        <p class="recommendation-price">${item.price}</p>
        <span class="recommendation-category">${item.category}</span>
      </div>
      <button class="add-to-cart-btn" onclick="addToCart('${item.name}')">Add</button>
    </div>
  `).join('');
}

// Populate shopping list
function populateShoppingList() {
  const container = document.getElementById('shopping-list');
  if (!container) return;
  
  container.innerHTML = Object.entries(groceryData.shoppingList).map(([category, items]) => `
    <div class="shopping-category">
      <h3 class="category-title">${category}</h3>
      <div class="category-items">
        ${items.map(item => `
          <div class="shopping-item ${item.checked ? 'checked' : ''}">
            <input type="checkbox" ${item.checked ? 'checked' : ''} onchange="toggleItem(this, '${item.name}')">
            <span class="item-name">${item.name}</span>
            <span class="item-quantity">${item.quantity}</span>
            <span class="item-price">${item.price}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

// Mock functions for interactions
function addToCart(itemName) {
  console.log(`Added ${itemName} to cart`);
  // In a real app, this would update the cart state
  alert(`Added ${itemName} to your cart!`);
}

function viewRecipe(recipeName) {
  console.log(`Viewing recipe: ${recipeName}`);
  // In a real app, this would open a recipe modal or navigate to recipe page
  alert(`Opening recipe for ${recipeName}!`);
}

function toggleItem(checkbox, itemName) {
  const itemElement = checkbox.closest('.shopping-item');
  if (checkbox.checked) {
    itemElement.classList.add('checked');
  } else {
    itemElement.classList.remove('checked');
  }
  console.log(`Toggled ${itemName}: ${checkbox.checked}`);
}

// Mock AI response function
async function getAIResponse() {
  const userInput = document.getElementById('user-input').value;
  const loadingElement = document.getElementById('chatbot-loading');
  const responseElement = document.getElementById('chatbot-response');
  
  if (!userInput.trim()) return;
  
  // Show loading
  loadingElement.style.display = 'block';
  responseElement.innerHTML = '';
  
  // Simulate AI response
  setTimeout(() => {
    loadingElement.style.display = 'none';
    
    const responses = [
      "Based on your preferences, I'd recommend trying our Quinoa Buddha Bowl recipe! It's packed with protein and perfect for your dietary needs.",
      "For healthy snacks, consider adding almonds, Greek yogurt, or hummus with vegetables to your cart. These align well with your nutrition goals.",
      "Looking at your meal history, you might enjoy our Mediterranean Chicken recipe. It's quick to prepare and uses ingredients you already love.",
      "I notice you're running low on protein sources. Would you like me to suggest some lean meat alternatives or plant-based options?"
    ];
    
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    
    responseElement.innerHTML = `
      <div class="ai-response">
        <div class="ai-avatar">🤖</div>
        <div class="ai-message">
          <p>${randomResponse}</p>
          <div class="ai-actions">
            <button onclick="addToCart('Suggested Item')" class="ai-action-btn">Add to Cart</button>
            <button onclick="viewRecipe('Suggested Recipe')" class="ai-action-btn">View Recipe</button>
          </div>
        </div>
      </div>
    `;
    
    document.getElementById('user-input').value = '';
  }, 1500);
}

// Mock functions for backend calls
async function fetchInsights() {
  return Promise.resolve([]);
}

async function updateUPFAndHarmfulIngredients() {
  return Promise.resolve();
}

async function fetchWeeklyScores() {
  return Promise.resolve([]);
}

async function displayUPFItemsOverlay() {
  return Promise.resolve();
}

async function displayHarmfulItemsOverlay() {
  return Promise.resolve();
}

async function getAIResponse() {
  return Promise.resolve();
}

