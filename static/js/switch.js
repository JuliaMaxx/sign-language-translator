let modelTypeBtn = document.getElementById('modelType');
let clearBtn = document.getElementById('clear');

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
    fetch('/clear', {method: 'POST'})
    .then(res => res.json())
})