document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.querySelector(".drop-zone");
    const fileInput = document.querySelector(".drop-zone__input");
    const form = document.getElementById("classifierForm");
    const prompt = document.querySelector(".drop-zone__prompt");

    if (!dropZone || !fileInput || !form) {
        console.error("Required elements not found");
        return;
    }

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('drop-zone--over');
    }

    function unhighlight() {
        dropZone.classList.remove('drop-zone--over');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files && files.length > 0) {
            fileInput.files = files;
            updateThumbnail(files[0]);
            form.submit();
        }
    }

    // Handle clicked upload
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
            updateThumbnail(this.files[0]);
            form.submit();
        }
    });

    function updateThumbnail(file) {
        if (!file.type.startsWith('image/')) {
            console.error('Not an image file');
            return;
        }

        const reader = new FileReader();
        
        reader.readAsDataURL(file);
        reader.onload = () => {
            if (prompt) {
                prompt.innerHTML = `
                    <i class="fas fa-spinner fa-spin" aria-hidden="true"></i>
                    <span>Processing image...</span>
                `;
            }
        };
    }
}); 