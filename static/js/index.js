const modelTypeBtn = document.getElementById('modelType');
const clearBtn = document.getElementById('clear');
const textDiv = document.getElementById("text");

const socket = io();

modelTypeBtn.addEventListener('click', () => {
    fetch('/switch', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.model == 'letters'){
            modelTypeBtn.textContent = "Numbers"
        } else {
            modelTypeBtn.textContent = "Letters"
        }
    });
});

clearBtn.addEventListener('click', () => {
    textDiv.innerText = "Start signing...";
    fetch('/clear', {method: 'POST'})
    .then(res => res.json())
});

socket.on('update_text', (data) => {
    textDiv.innerText = data.text;
});