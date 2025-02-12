
#####################
####################

import streamlit as st
import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import uuid 
# 📌 Load AI Model
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
        activation=None
    )
    model.load_state_dict(torch.load("satellite_segmentation.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# 📌 Preprocessing Function
def preprocess_image(image):
    transform = Compose([
        Resize(512, 512),
        Normalize(),
        ToTensorV2()
    ])
    image = np.array(image)
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)  # Add batch dimension

# 📌 AI Prediction Function - Fixing Color Mapping
def predict_mask(image):
    image_tensor = preprocess_image(image)  # Assuming this function exists
    with torch.no_grad():
        pred_mask = model(image_tensor)
        pred_mask = torch.argmax(pred_mask.squeeze(0), dim=0).cpu().numpy()
        image_np = np.array(image)  # Convert PIL image to NumPy array

    # Ensure the mask is in proper shape (512, 512)
    if len(pred_mask.shape) == 2:
        rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

        # Assign colors properly
        rgb_mask[pred_mask == 1] = [255, 0, 0]   # Red (Solar Panel)
        rgb_mask[pred_mask == 2] = [0, 255, 0]   # Green (Body)
        rgb_mask[pred_mask == 3] = [0, 0, 255]   # Blue (Antenna)

        return rgb_mask
    else:
        st.error("AI mask is not in expected format.")
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
# Overlay Function
def overlay_mask(image, mask, alpha=0.5):
    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != image.size[::-1]:  # Fixing shape comparison (image.size is (width, height))
        mask = cv2.resize(mask, image.size)  # Resize mask to match image dimensions

    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Blend the original image with the mask
    blended = cv2.addWeighted(image_np, 1 - alpha, mask, alpha, 0)
    return blended

# 📌 IoU Calculation in RGB Space
def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection / union) * 100 if union != 0 else 0

# 📌 Color Mapping for Three Objects
color_mapping = {
    "Red (Object 1)": "#FF0000",
    "Green (Object 2)": "#00FF00",
    "Blue (Object 3)": "#0000FF"
}


# 🎮 Streamlit UI
st.title("🚀 Satellite Masking Game")
st.write("Click and draw on the image to create your mask. AI will compare it!")

# 📜 Instructions
st.markdown("### 📜 Instructions")
st.write("""
1. Use the **drawing canvas below** to draw a mask on the satellite.
2. Choose a color (Red- Solar panel, Green- Body, or Blue- Antenna) to identify different objects.
3. If you make a mistake, click **'Reset Mask'** to start over.
4. Click **'Submit Mask'** to let AI compare your mask.
5. AI will check **accuracy using IoU**.
6. If **IoU > 80%**, you **win! 🚀**
""")

# 📌 Load Satellite Image
image_path = r"img_resize_31.png" # Change this to your dataset path
satellite_image = Image.open(image_path).resize((512, 512))

# 📌 Initialize Session State for Mask Storage
if "user_mask" not in st.session_state:
    st.session_state.user_mask = Image.new("RGB", (512, 512), (0, 0, 0))

# 📌 Display Satellite Image
# st.write("### 🛰️ Satellite Image")
# st.image(satellite_image, caption="Satellite Image", use_container_width=True)

# 📌 Color Selection
selected_color = st.radio("Select a color to draw the mask:", list(color_mapping.values()), format_func=lambda c: f"🖌️ {c}")

# 📌 Initialize Session State
if "user_mask" not in st.session_state:
    st.session_state.user_mask = Image.new("RGB", (512, 512), (0, 0, 0))
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = str(uuid.uuid4())  # Unique key for the canvas



# 🖌️ **Drawing Area for Player Mask**
st.write("### 🖌️ Draw Your Mask Below")
canvas_result = st_canvas(
    fill_color=selected_color + "30",  # Transparent fill with selected color
    stroke_width=5,
    stroke_color=selected_color,  # Player uses selected color for mask
    background_image=satellite_image,  # Display satellite image
    height=512,
    width=512,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key
)
# 📌 Reset Button Logic
# 📌 Reset Button Logic
if st.button("🔄 Reset Mask"):
    st.session_state.user_mask = Image.new("RGB", (512, 512), (0, 0, 0))  # Clear the mask
    st.session_state.canvas_key = str(uuid.uuid4())  # Generate a new key for a full reset

# 📌 Submit Button
if st.button("✅ Submit Mask"):
    if canvas_result.image_data is not None:
        st.write("### ✅ Mask Submitted! AI is verifying...")

        # Convert player's mask from canvas to NumPy array (RGB format)
        player_mask = np.array(canvas_result.image_data[:, :, :3])  # Extract RGB channels

        # AI Predicts Mask
        ai_mask = predict_mask(satellite_image)
        overlay_image = overlay_mask(satellite_image, ai_mask, alpha=0.4)  # Adjust alpha for transparency

        # IoU Scores for Each Object Color
        iou_scores = {}

        for idx, (label, color) in enumerate(color_mapping.items()):
            # Extract the specific color from the player mask (RGB comparison)
            color_rgb = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])  # Convert Hex to RGB
            color_channel = np.all(player_mask == color_rgb, axis=-1).astype(np.uint8)  # Find pixels matching color
            
            # Extract the same object from AI prediction
            ai_color_channel = np.all(ai_mask == color_rgb, axis=-1).astype(np.uint8)

            # IoU Calculation
            iou_scores[label] = calculate_iou(color_channel, ai_color_channel)

        # 📌 Display IoU Scores
        st.write("🎯 **IoU Scores**")
        for label, iou in iou_scores.items():
            st.write(f"**{label}:** {iou:.2f}%")

        # 📌 Display Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(player_mask, caption="🎨 Your Mask")
        with col2:
            st.image(ai_mask, caption="🤖 AI Predicted Mask")
        with col3:
            st.image(overlay_image, caption="🔍 Overlay Image")

        # 📌 Feedback Based on IoU Score
        if all(iou > 80 for iou in iou_scores.values()):
            st.success("🚀 Perfect Masking! Your satellite is launched!")
        else:
            st.error("⚠️ Your mask needs improvements! Try again.")
    else:
        st.warning("❗ Please draw your mask before submitting.")

