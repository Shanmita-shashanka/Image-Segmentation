async function uploadImage() {
    console.log("Button clicked");

    const fileInput = document.getElementById("imageInput");
    const outputImage = document.getElementById("outputImage");

    if (!fileInput.files.length) {
        alert("Please select an image first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:8000/segment", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Convert hex to bytes
        const bytes = new Uint8Array(
            data.mask.match(/.{1,2}/g).map(b => parseInt(b, 16))
        );

        const blob = new Blob([bytes], { type: "image/png" });
        outputImage.src = URL.createObjectURL(blob);

    } catch (error) {
        console.error("Error:", error);
        alert("Segmentation failed. Check backend.");
    }
}