let modelTypeSpan = document.getElementById('modelType');

fetch('/model')
      .then(res => res.json())
      .then(data => modelTypeSpan.textContent = `${data.model} mode`);

    document.addEventListener('keydown', function(event) {
    if (event.key === 'n') {
        fetch('/switch', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            modelTypeSpan.textContent = `${data.model} mode`;
        });
    }
    });