async function fetchData() {
    const symbol = document.getElementById('symbolInput').value.toUpperCase();
    if (!symbol) {
      alert('Please enter a stock symbol');
      return;
    }
  
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/stock/${symbol}`);
      const data = await response.json();
      const timeSeries = data['Time Series (5min)'];
  
      if (timeSeries) {
        const dates = Object.keys(timeSeries);
        const stockData = dates.map(date => parseFloat(timeSeries[date]['1. open']));
  
        const chartData = {
          labels: dates,
          datasets: [
            {
              label: `${symbol} Stock Price`,
              data: stockData,
              fill: false,
              borderColor: 'rgb(75, 192, 192)',
              tension: 0.1
            }
          ]
        };
  
        const ctx = document.getElementById('stockChart').getContext('2d');
        new Chart(ctx, {
          type: 'line',
          data: chartData,
          options: {
            scales: {
              x: {
                reverse: true,
              }
            }
          }
        });
      } else {
        alert('No data found for the given symbol');
      }
    } catch (error) {
      console.error('Error fetching stock data:', error);
      alert('Failed to fetch stock data');
    }
  }
  