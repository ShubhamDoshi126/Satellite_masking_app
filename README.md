## Inspiration
Satellite imagery is crucial for many applications, from environmental monitoring to urban planning. However, labeling and segmenting objects in satellite images can be challenging. We wanted to create an interactive AI-powered game that makes learning about image segmentation fun while helping users understand how AI models work in object recognition.

## What it does
The Satellite Masking Game is an interactive AI-powered segmentation game where players:

Draw segmentation masks on satellite images using a color-coded system.
The AI model predicts the correct segmentation mask.
The game compares the user’s mask with the AI-generated mask using IoU (Intersection over Union) to calculate accuracy.
If IoU > 80%, the user wins! 🚀

## How we built it
Frontend & UI: Built using Streamlit, which provides an intuitive and interactive experience.
AI Model: Implemented a U-Net segmentation model with a ResNet34 encoder, trained to recognize different parts of satellite images.
Image Processing: Used OpenCV, PIL, and Albumentations for image transformations and augmentation.
Cloud Deployment: Hosted on Huggingface spaces

## Challenges we ran into
Model Deployment: The model file was too large for GitHub, so couldn't upload the model on it.
IoU Calculation: Ensuring the mask comparison was fair and accurate while considering different drawing styles.
UI Optimization: Integrating streamlit-drawable-canvas for a smooth user experience. That is one of the issue we are facing to be displayed on huggingface spaces

## Accomplishments that we're proud of
✅ Successfully built an interactive AI game that blends deep learning with an engaging UI.
✅ Implemented real-time IoU scoring, allowing players to instantly compare their segmentation with AI predictions.
✅ Created a fun and educational AI experience that can be used for learning about image segmentation.

## What we learned
What we learned
📌 Deploying AI models on Streamlit Cloud, Huggingface spaces and other options and handling large model files efficiently.
📌 Optimizing UI & performance when working with real-time image segmentation.
📌 Improving AI-human interaction, allowing users to intuitively engage with machine learning models.
📌 Cloud-based game development, making AI-powered applications accessible to everyone.

## What's next for Satellite masking game
🚀 Expand the dataset: Add more diverse satellite images for segmentation and the objects to not be limited to 3 but more detailed and important information about satellite.
🎨 Improve user interaction: Add more drawing tools and brush options.
🤖 Fine-tune the AI model: Improve segmentation accuracy and adapt it for more complex objects.
📡 Use real satellite images from NASA or OpenStreetMap for real-world use cases.

## Hosted on
https://huggingface.co/spaces/Shubham126/Satellite_masking_game
