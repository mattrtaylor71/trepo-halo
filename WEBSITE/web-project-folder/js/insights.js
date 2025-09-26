async function fetchInsights() {
    const scoreNumberElement = document.querySelector(".score-number");
    const scoreStatusElement = document.querySelector(".score-status");
    const scoreCircleElement = document.querySelector(".score-circle");
    const tipsHeader = document.querySelector(".tips-section h2");
    const tipsSection = document.querySelector(".tips-section p"); // Add selector for tips-section paragraph
    const hydrationTipSection = document.querySelector(".hydration-tip"); // Add selector for hydration-tip

    // Function to determine ranking based on score
    function getRanking(score) {
        if (score >= 90) return { text: "Exceptional", color: "#28a745" }; // Green
        if (score >= 80) return { text: "Optimal", color: "#007bff" }; // Blue
        if (score >= 70) return { text: "Excellent", color: "#17a2b8" }; // Cyan
        if (score >= 60) return { text: "Good", color: "#ffc107" }; // Yellow
        if (score >= 50) return { text: "Fair", color: "#fd7e14" }; // Orange
        if (score >= 30) return { text: "Needs Improvement", color: "#dc3545" }; // Red
        if (score >= 10) return { text: "Poor", color: "#6c757d" }; // Grey
        return { text: "Critical", color: "#343a40" }; // Dark Grey
    }

    try {
        // Fetch the web_id from local storage (Auth0 stored user data)
        const user = localStorage.getItem("user");
        console.log("Retrieved user from localStorage:", user);

        const webId = user ? JSON.parse(user).sub : null;
        console.log("Parsed webId:", webId);

        if (!webId) {
            throw new Error("User not authenticated. Web ID missing.");
        }

        // Fetch data from the /fetchInsights endpoint
        console.log("Sending GET request to /fetchInsights with webId:", webId);
        const response = await fetch(`https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${webId}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
        });

        console.log("API response status:", response.status);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Fetched data from /fetchInsights:", data);

        if (!data || data.length === 0) {
            console.warn("No data found for the given webId.");
            scoreNumberElement.textContent = "N/A"; // Handle no data
            scoreStatusElement.textContent = "No Data"; // Display status
            scoreStatusElement.style.color = "#6c757d"; // Grey
            scoreCircleElement.style.borderColor = "#6c757d"; // Grey
            tipsHeader.style.color = "#6c757d"; // Grey
            tipsSection.textContent = "No insights available."; // Handle empty insights
            hydrationTipSection.textContent = "No hydration tips available."; // Handle empty tips
            return;
        }

        // Find the most recent score using the _updatedDate field
        const mostRecentScore = data.reduce((latest, item) => {
            const itemDate = new Date(item._updatedDate);
            return (!latest || itemDate > new Date(latest._updatedDate)) ? item : latest;
        }, null);

        console.log("Most recent score entry:", mostRecentScore);

        // Display the most recent score, ranking, reasoning, and recommendation
        if (mostRecentScore) {
            const roundedScore = Math.round(mostRecentScore.score || 0);
            const ranking = getRanking(roundedScore);

            console.log("Displaying rounded score:", roundedScore);
            console.log("Displaying ranking:", ranking);

            // Update score and ranking
            scoreNumberElement.textContent = roundedScore; // Show rounded score
            scoreStatusElement.textContent = ranking.text; // Show ranking

            // Apply color coding
            scoreCircleElement.style.borderColor = ranking.color; // Circle border color
            scoreStatusElement.style.color = ranking.color; // Score status color
            tipsHeader.style.color = ranking.color; // Tips header color

            // Update reasoning and recommendation
            const reasoning = mostRecentScore.reasoning || "No reasoning available.";
            const recommendation = mostRecentScore.recommendation || "No recommendation available.";

            console.log("Displaying reasoning:", reasoning);
            console.log("Displaying recommendation:", recommendation);

            tipsSection.textContent = reasoning; // Show reasoning in tips-section
            hydrationTipSection.textContent = recommendation; // Show recommendation in hydration-tip
        } else {
            console.warn("No valid score or insights found in the most recent entry.");
            scoreNumberElement.textContent = "N/A"; // Handle no valid score
            scoreStatusElement.textContent = "No Data"; // Display status
            scoreStatusElement.style.color = "#6c757d"; // Grey
            scoreCircleElement.style.borderColor = "#6c757d"; // Grey
            tipsHeader.style.color = "#6c757d"; // Grey
            tipsSection.textContent = "No insights available."; // Handle empty insights
            hydrationTipSection.textContent = "No hydration tips available."; // Handle empty tips
        }
    } catch (error) {
        console.error("Error fetching insights:", error);
        scoreNumberElement.textContent = "Error"; // Display error message
        scoreStatusElement.textContent = "Error"; // Display status
        scoreStatusElement.style.color = "#6c757d"; // Grey
        scoreCircleElement.style.borderColor = "#6c757d"; // Grey
        tipsHeader.style.color = "#6c757d"; // Grey
        tipsSection.textContent = "Unable to fetch insights."; // Handle API error
        hydrationTipSection.textContent = "Unable to fetch hydration tips."; // Handle API error
    }
}

// Initialize data fetch
window.onload = fetchInsights;
