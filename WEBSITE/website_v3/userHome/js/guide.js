// Simplified guide.js for local development
// Removes all backend API dependencies and uses mock data

/**********************************************
 * MOCK DATA
 **********************************************/
let mockGuideData = {
    steps: [
        {
            id: 1,
            title: "Download the App",
            description: "Get the TREPO app from your device's app store",
            completed: true,
            image: "img/image-6.png"
        },
        {
            id: 2,
            title: "Create Your Account",
            description: "Sign up with your email and create a profile",
            completed: true,
            image: "img/image-7.png"
        },
        {
            id: 3,
            title: "Mount Your Scanner",
            description: "Attach the scanner to your kitchen counter or wall",
            completed: false,
            image: "img/image-8.png"
        },
        {
            id: 4,
            title: "Start Scanning",
            description: "Begin scanning your food items to track consumption",
            completed: false,
            image: "img/image-9.png"
        },
        {
            id: 5,
            title: "View Your Insights",
            description: "Check your dashboard for personalized recommendations",
            completed: false,
            image: "img/image-11.png"
        }
    ],
    tips: [
        "Keep your scanner clean for best results",
        "Scan items before opening them",
        "Check your insights regularly for trends",
        "Use the shopping list feature to plan meals"
    ]
};

/**********************************************
 * FETCH GUIDE DATA (MOCK)
 **********************************************/
async function fetchGuideData() {
    console.log("Fetching mock guide data...");
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    return mockGuideData;
}

/**********************************************
 * RENDER GUIDE STEPS
 **********************************************/
async function renderGuideSteps() {
    const container = document.getElementById("guide-steps-container");
    if (!container) return;
    
    const data = await fetchGuideData();
    
    container.innerHTML = `
        <div class="guide-header">
            <h2>Setup Guide</h2>
            <p>Follow these steps to get started with TREPO</p>
        </div>
        <div class="steps-list">
            ${data.steps.map((step, index) => `
                <div class="step-item ${step.completed ? 'completed' : ''}">
                    <div class="step-number">${index + 1}</div>
                    <div class="step-content">
                        <div class="step-image">
                            <img src="${step.image}" alt="${step.title}">
                        </div>
                        <div class="step-info">
                            <h3 class="step-title">${step.title}</h3>
                            <p class="step-description">${step.description}</p>
                            ${step.completed ? '<span class="step-status">✓ Completed</span>' : '<span class="step-status pending">Pending</span>'}
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**********************************************
 * RENDER TIPS
 **********************************************/
async function renderTips() {
    const container = document.getElementById("tips-container");
    if (!container) return;
    
    const data = await fetchGuideData();
    
    container.innerHTML = `
        <div class="tips-section">
            <h3>Pro Tips</h3>
            <div class="tips-list">
                ${data.tips.map(tip => `
                    <div class="tip-item">
                        <span class="tip-icon">💡</span>
                        <span class="tip-text">${tip}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**********************************************
 * MARK STEP COMPLETE
 **********************************************/
async function markStepComplete(stepId) {
    console.log(`Marking step ${stepId} as complete...`);
    
    const step = mockGuideData.steps.find(s => s.id === stepId);
    if (step) {
        step.completed = true;
        await renderGuideSteps(); // Re-render to show updated state
        console.log(`Step ${stepId} marked as complete`);
    }
}

/**********************************************
 * INITIALIZE PAGE
 **********************************************/
document.addEventListener("DOMContentLoaded", async () => {
    console.log("Initializing guide page...");
    
    // Render all sections
    await renderGuideSteps();
    await renderTips();
    
    // Set up event listeners for step completion
    const stepItems = document.querySelectorAll(".step-item");
    stepItems.forEach((item, index) => {
        item.addEventListener("click", () => {
            const stepId = index + 1;
            markStepComplete(stepId);
        });
    });
    
    console.log("Guide page initialized successfully");
});
