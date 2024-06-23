document.addEventListener('DOMContentLoaded', function () {
    fetchHoldings();
    fetchTransactions();
});

function fetchHoldings() {
    fetch('http://127.0.0.1:5000/api/holdings')
        .then(response => response.json())
        .then(data => {
            let holdingsBody = document.getElementById('holdings-body');
            let totalBalance = 0;
            data.forEach(holding => {
                let row = holdingsBody.insertRow();
                row.insertCell(0).innerText = holding.symbol;
                row.insertCell(1).innerText = holding.shares;
                row.insertCell(2).innerText = holding.purchase_price;
                row.insertCell(3).innerText = holding.current_price;
                row.insertCell(4).innerText = holding.market_value.toFixed(2);
                row.insertCell(5).innerText = holding.unrealized_gain.toFixed(2);
                totalBalance += holding.market_value;
            });
            document.getElementById('total-balance').innerText = totalBalance.toFixed(2);
            // Calculate overall performance (placeholder logic)
            document.getElementById('overall-performance').innerText = ((totalBalance - 10000) / 10000 * 100).toFixed(2);
        })
        .catch(error => console.error('Error fetching holdings:', error));
}

function fetchTransactions() {
    fetch('http://127.0.0.1:5000/api/transactions')
        .then(response => response.json())
        .then(data => {
            let transactionsBody = document.getElementById('transactions-body');
            data.forEach(transaction => {
                let row = transactionsBody.insertRow();
                row.insertCell(0).innerText = transaction.date;
                row.insertCell(1).innerText = transaction.type;
                row.insertCell(2).innerText = transaction.symbol;
                row.insertCell(3).innerText = transaction.shares;
                row.insertCell(4).innerText = transaction.price;
                row.insertCell(5).innerText = (transaction.shares * transaction.price).toFixed(2);
                row.insertCell(6).innerText = transaction.fees;
            });
        })
        .catch(error => console.error('Error fetching transactions:', error));
}
