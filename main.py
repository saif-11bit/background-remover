from flask import Flask, request, jsonify, url_for
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = YOLO("yolo11n-seg.pt")

@app.route("/segment", methods=["POST"])
def segment_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    image_bytes = file.read()
    image = np.frombuffer(image_bytes, np.uint8)  # Convert to NumPy array
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image (e.g., BGR format)

    # Pass the image to the model
    result = model(image)
    mask_tensor = result[0].masks.data[0].cpu()  # Move mask to CPU if needed
    mask = mask_tensor.numpy()

    mask = cv2.resize(mask, (result[0].orig_shape[1], result[0].orig_shape[0]))

    orig_img = result[0].orig_img

    mask = (mask > 0).astype(np.uint8) * 255

    # Create an alpha channel based on the mask
    b, g, r = cv2.split(orig_img)  # Split original image into channels
    rgba = cv2.merge((b, g, r, mask))  # Add the mask as the alpha channel

    # Save the segmented part with transparency
    output_path = "static/segmented_part_test_2.png"
    cv2.imwrite(output_path, rgba)

    return jsonify({
        "foreground_url": url_for("static", filename="segmented_part_test_2.png", _external=True),
    })



if __name__ == "__main__":
    app.run(debug=True)
