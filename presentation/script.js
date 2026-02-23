document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    let currentSlide = 0;

    function showSlide(index) {
        // Handle wrap-around or bounds
        if (index < 0) index = 0;
        if (index >= slides.length) index = slides.length - 1;

        // Remove active class from all slides
        slides.forEach(slide => {
            slide.classList.remove('active');
        });

        // Add active class to current slide
        slides[index].classList.add('active');
        currentSlide = index;

        // Update button states
        prevBtn.disabled = currentSlide === 0;
        nextBtn.innerHTML = currentSlide === slides.length - 1 ? 'Finish' : 'Next';
        
        // Optional: fade out 'Finish' button if needed, but for now just keep as 'Next/Finish'
        if (currentSlide === 0) {
            prevBtn.style.opacity = '0.5';
            prevBtn.style.pointerEvents = 'none';
        } else {
            prevBtn.style.opacity = '1';
            prevBtn.style.pointerEvents = 'auto';
        }
    }

    prevBtn.addEventListener('click', () => {
        if (currentSlide > 0) {
            showSlide(currentSlide - 1);
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentSlide < slides.length - 1) {
            showSlide(currentSlide + 1);
        }
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {
            if (currentSlide < slides.length - 1) showSlide(currentSlide + 1);
        } else if (e.key === 'ArrowLeft') {
            if (currentSlide > 0) showSlide(currentSlide - 1);
        }
    });

    // Initialize
    showSlide(0);
});
