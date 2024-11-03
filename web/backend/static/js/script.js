document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.querySelector('form');

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Handle form submission logic here
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });
        
        // Fetch prediction
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            predict = document.getElementById("predict")
            predict.innerHTML = `The prediction is: ${data["prediction"]}`
        })
        .catch(error => console.error('Error:', error));
    });
});

