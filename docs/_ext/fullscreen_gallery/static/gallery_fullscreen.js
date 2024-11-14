document.addEventListener('DOMContentLoaded', function() {
    // Create modal container
    const modal = document.createElement('div');
    modal.className = 'gallery-fullscreen-modal';
    modal.innerHTML = `
        <button class="gallery-fullscreen-close"><span>×</span></button>
        <div class="gallery-fullscreen-content"></div>
    `;
    document.body.appendChild(modal);

    // Add click handlers to all fullscreen buttons
    document.querySelectorAll('.gallery-fullscreen-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const wrapper = this.closest('.gallery-fullscreen-wrapper');
            const content = wrapper.querySelector('img, div');

            if (content) {
                const clone = content.cloneNode(true);
                modal.querySelector('.gallery-fullscreen-content').innerHTML = '';
                modal.querySelector('.gallery-fullscreen-content').appendChild(clone);
                modal.classList.add('active');
            }
        });
    });

    // Close modal on click outside or close button
    modal.addEventListener('click', function(e) {
        if (e.target === modal || e.target.closest('.gallery-fullscreen-close')) {
            modal.classList.remove('active');
        }
    });

    // Close modal on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            modal.classList.remove('active');
        }
    });
});