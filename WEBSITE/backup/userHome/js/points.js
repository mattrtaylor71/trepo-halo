async function fetchAndDisplayPoints() {
  try {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) {
      console.warn("User not logged in. Skipping points fetch.");
      return 0;
    }

    const webId = user.sub;
    const response = await fetch(`https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${webId}`);
    if (!response.ok) throw new Error("Failed to fetch insights.");

    const data = await response.json();
    if (!data.length) return 0;

    const mostRecent = data.reduce((latest, item) => {
      const currentDate = new Date(item._createdDate);
      return !latest || currentDate > new Date(latest._createdDate) ? item : latest;
    }, null);

    if (mostRecent && mostRecent.points != null) {
      const points = parseInt(mostRecent.points);
      console.log(`🏆 Points: ${points}`);

      const pointsElement = document.querySelector(".user-points");
      if (pointsElement) {
        pointsElement.textContent = points;
      }

      return points;
    } else {
      return 0;
    }
  } catch (err) {
    console.error("Error fetching user points:", err);
    return 0;
  }
}
