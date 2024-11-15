let scrollPosition = 0;

function openModal(modal) {
    scrollPosition = window.pageYOffset;
    document.body.style.overflow = 'hidden';
    document.body.style.position = 'fixed';
    document.body.style.top = `-${scrollPosition}px`;
    document.body.style.width = '100%';
    modal.classList.add('active');
}

function closeModal(modal) {
    document.body.style.removeProperty('overflow');
    document.body.style.removeProperty('position');
    document.body.style.removeProperty('top');
    document.body.style.removeProperty('width');
    window.scrollTo(0, scrollPosition);
    modal.classList.remove('active');
}


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
                openModal(modal);
            }
        });
    });

    // Close modal on click outside or close button
    modal.addEventListener('click', function(e) {
        if (e.target === modal || e.target.closest('.gallery-fullscreen-close')) {
            closeModal(modal);
        }
    });

    // Close modal on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal(modal);
        }
    });
});