async function fetchInsights() {
    const averageScoreElement = document.getElementById("average-score");
    const highlightsContainer = document.getElementById("highlights-container");
    const lowlightsContainer = document.getElementById("lowlights-container");

    try {
        // Fetch data from backend
        const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_name: JSON.parse(localStorage.getItem("user")).sub })
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();

        // Filter and calculate score
        const last7DaysData = data.filter(item => {
            const itemDate = new Date(item._createdDate);
            const sevenDaysAgo = new Date();
            sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
            return itemDate >= sevenDaysAgo;
        });

        const scores = last7DaysData.map(item => parseFloat(item.score) || 0);
        const averageScore = scores.reduce((a, b) => a + b, 0) / scores.length || 0;
        averageScoreElement.textContent = `${averageScore.toFixed(1)}%`;

        // Sort items by score
        const sortedData = last7DaysData.sort((a, b) => b.score - a.score);

        // Display highlights
        const highlights = sortedData.slice(0, 3);
        highlights.forEach(item => {
            const div = document.createElement("div");
            div.className = "item-box";
            div.innerHTML = `
                <img src="${item.images}" alt="${item.title}">
                <p class="item-title">${item.title}</p>
                <p class="item-score">Score: ${item.score}</p>
            `;
            highlightsContainer.appendChild(div);
        });

        // Display lowlights
        const lowlights = sortedData.slice(-3);
        lowlights.forEach(item => {
            const div = document.createElement("div");
            div.className = "item-box";
            div.innerHTML = `
                <img src="${item.images}" alt="${item.title}">
                <p class="item-title">${item.title}</p>
                <p class="item-score">Score: ${item.score}</p>
            `;
            lowlightsContainer.appendChild(div);
        });
    } catch (error) {
        console.error("Error fetching insights:", error);
        document.querySelector(".container").innerHTML = `<p>Failed to load insights. Please try again later.</p>`;
    }
}

// Initialize data fetch
window.onload = fetchInsights;
