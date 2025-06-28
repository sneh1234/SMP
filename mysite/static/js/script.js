// Function to fetch recommended stocks from CSV file
function fetchRecommendedStocks() {
    fetch('/static/js/recommendation.csv')
    .then(response => response.text())
    .then(data => {
        const rows = data.split('\n').slice(1, 11); // Skip header row
        const tbody = document.querySelector('#recommended-stocks tbody');
        rows.forEach(row => {
            const [stock, expectedReturn] = row.split(','); // Assuming first column contains stock names and second column contains expected returns
            const tr = document.createElement('tr');
            if (expectedReturn > 0.13){
                tr.className = "HighReco"}
            tr.innerHTML = `<td>${stock}</td><td>${Math.round(expectedReturn * 10000)/100}</td>`;
            tbody.appendChild(tr);
        });
    })
    .catch(error => console.error('Error fetching recommended stocks:', error));
}

// Example JavaScript code for updating the exit notification
// You can use your preferred method to dynamically update the notification based on certain conditions



function fetchExitList() {
    fetch('/static/js/archives.csv')
    .then(response => response.text())
    .then(data => {
        const rows = data.split('\n')
        const lastline = rows[rows.length-2].split(',')
        const stocklist = lastline[lastline.length - 3]
        const exitList = stocklist.split(':')
        const exitListElement = document.getElementById('exit-list');
        if (stocklist.length > 0) {
            exitListElement.innerHTML = `<strong>Notification:</strong> Exit from the following stocks is required: ${exitList.join(', ')}.`;
        } else {
            exitListElement.innerHTML = "<strong>Notification:</strong> No exit required at the moment.";
        }
    })
    .catch(error => console.error('Error fetching exit list:', error));
}

// Call functions to set up the page
fetchRecommendedStocks();
fetchExitList();

