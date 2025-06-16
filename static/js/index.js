const modelTypeBtn = document.getElementById('modelType');
const modelLanguageBtn = document.getElementById('modelLanguage');
const clearBtn = document.getElementById('clear');
const textDiv = document.getElementById("text");

const socket = io();

modelTypeBtn.addEventListener('click', () => {
    fetch('/switch', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.type == 'letters'){
            modelTypeBtn.textContent = "Numbers"
        } else {
            modelTypeBtn.textContent = "Letters"
        }
    });
});

modelLanguageBtn.addEventListener('click', () => {
    fetch('/toggle_language', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.language == 'bsl'){
            modelLanguageBtn.textContent = "ASL"
        } else {
            modelLanguageBtn.textContent = "BSL"
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