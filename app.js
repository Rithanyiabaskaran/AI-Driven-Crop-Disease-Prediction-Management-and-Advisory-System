const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

if (imageInput && previewImage) {
  imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        previewImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  });
}