# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import uuid
import os
from io import BytesIO


# Set page config with wider layout and custom theme
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with sticky footer
st.markdown("""
<style>
    /* Global styles for layout */
    .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Prediction box */
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 16px 0;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .prediction-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton>button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 14px;
        transition: background 0.2s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background: #1e40af;
        transform: translateY(-2px);
    }

.footer-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

@media (min-width: 768px) {
  .footer-content {
    flex-direction: row;
    justify-content: space-between;
  }
}

 .footer-content h1 {
  margin-bottom: 0px;
    text-align: center;
}


.social-links {
  margin-top: 15px;
}

.footer-link {
  color: #e0e0e0;
  font-size: 15px;
  margin-right: 15px;
  text-decoration: none;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  transition: color 0.3s ease, transform 0.3s ease;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
  transition: color 0.3s ease;
}

.footer-link:hover {
  color: #1da1f2;
}

.contact-list {
  list-style: none;
  padding: 0;
  font-size: 14px;
}

.contact-list li {
  margin: 0px 0;
  display: flex;
  align-items: center;
}

.contact-list i {
  margin-right: 10px;
  color: #1da1f2;
}

/* Crop styling */
.crop-container {
    position: relative;
    margin-bottom: 20px;
}

.crop-preview {
    max-width: 100%;
    border: 2px dashed #ccc;
    border-radius: 8px;
}

.crop-controls {
    margin-top: 10px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.crop-result {
    margin-top: 20px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
}

.crop-section {
    margin-bottom: 30px;
    padding: 20px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.original-image {
    max-width: 400px; /* Set your desired max width */
    height: auto; /* Maintain aspect ratio */
    display: block;
    margin: 0 auto; /* Center the image *
}
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_custom_cnn.h5')

model = load_model()

# Enhanced class labels with descriptions
class_labels = {
    0: {'name': 'Speed Limit 20 kmph', 'description': 'Maximum speed allowed is 20 kilometers per hour'},
    1: {'name': 'Speed Limit 30 kmph', 'description': 'Maximum speed allowed is 30 kilometers per hour'},
    2: {'name': 'Speed Limit 50 kmph', 'description': 'Maximum speed allowed is 50 kilometers per hour'},
    3: {'name': 'Speed Limit 60 kmph', 'description': 'Maximum speed allowed is 60 kilometers per hour'},
    4: {'name': 'Speed Limit 70 kmph', 'description': 'Maximum speed allowed is 70 kilometers per hour'},
    5: {'name': 'Speed Limit 80 kmph', 'description': 'Maximum speed allowed is 80 kilometers per hour'},
    6: {'name': 'End of Speed Limit 80 kmph', 'description': 'The 80 kmph speed limit zone ends here'},
    7: {'name': 'Speed Limit 100 kmph', 'description': 'Maximum speed allowed is 100 kilometers per hour'},
    8: {'name': 'Speed Limit 120 kmph', 'description': 'Maximum speed allowed is 120 kilometers per hour'},
    9: {'name': 'No Passing', 'description': 'No overtaking other vehicles in this zone'},
    10: {'name': 'No Passing for Vehicles Over 3.5 Tons', 'description': 'Prohibits vehicles over 3.5 tons from passing others'},
    11: {'name': 'Right-of-Way at Intersection', 'description': 'Drivers must yield to vehicles approaching from the right at the next intersection'},
    12: {'name': 'Priority Road', 'description': 'Indicates the road has right-of-way at intersections'},
    13: {'name': 'Yield', 'description': 'Drivers must slow down and yield to traffic on the intersecting road'},
    14: {'name': 'Stop', 'description': 'Complete stop required before proceeding'},
    15: {'name': 'No Vehicles', 'description': 'No vehicles of any kind allowed beyond this point'},
    16: {'name': 'Vehicles Over 3.5 Tons Prohibited', 'description': 'No vehicles heavier than 3.5 tons allowed'},
    17: {'name': 'No Entry', 'description': 'Entry forbidden for all vehicles'},
    18: {'name': 'General Caution', 'description': 'Warning of potential hazards ahead'},
    19: {'name': 'Dangerous Curve Left', 'description': 'Sharp curve to the left ahead'},
    20: {'name': 'Dangerous Curve Right', 'description': 'Sharp curve to the right ahead'},
    21: {'name': 'Double Curve', 'description': 'Series of two curves ahead (first left, then right)'},
    22: {'name': 'Bumpy Road', 'description': 'Warning of uneven road surface ahead'},
    23: {'name': 'Slippery Road', 'description': 'Road may be slippery when wet'},
    24: {'name': 'Road Narrows on Right', 'description': 'Road narrows on the right side ahead'},
    25: {'name': 'Road Work', 'description': 'Construction or maintenance work ahead'},
    26: {'name': 'Traffic Signals', 'description': 'Traffic lights ahead'},
    27: {'name': 'Pedestrians', 'description': 'Pedestrian crossing area ahead'},
    28: {'name': 'Children Crossing', 'description': 'Warning of children likely crossing road ahead'},
    29: {'name': 'Bicycles Crossing', 'description': 'Warning of bicycle crossing area ahead'},
    30: {'name': 'Beware of Ice/Snow', 'description': 'Warning of potential ice or snow on road'},
    31: {'name': 'Wild Animals Crossing', 'description': 'Warning of animals potentially crossing road'},
    32: {'name': 'End of All Speed and Passing Limits', 'description': 'All previous speed and passing restrictions end here'},
    33: {'name': 'Turn Right Ahead', 'description': 'Mandatory right turn ahead'},
    34: {'name': 'Turn Left Ahead', 'description': 'Mandatory left turn ahead'},
    35: {'name': 'Ahead Only', 'description': 'Drivers must continue straight ahead'},
    36: {'name': 'Go Straight or Right', 'description': 'Drivers may go straight or turn right'},
    37: {'name': 'Go Straight or Left', 'description': 'Drivers may go straight or turn left'},
    38: {'name': 'Keep Right', 'description': 'Drivers must keep to the right'},
    39: {'name': 'Keep Left', 'description': 'Drivers must keep to the left'},
    40: {'name': 'Roundabout Mandatory', 'description': 'Warning of roundabout ahead, traffic must circulate counter-clockwise'},
    41: {'name': 'End of No Passing', 'description': 'No passing zone ends here'},
    42: {'name': 'End of No Passing for Vehicles Over 3.5 Tons', 'description': 'No passing restriction for heavy vehicles ends here'}
}

# Image preprocessing function
def load_and_preprocess_image(img):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (48, 48))
    return image / 255.0

# Function to analyze a cropped image
def analyze_cropped_image(crop, crop_name):
    with st.spinner(f'🔍 Analyzing {crop_name}...'):
        processed_image = load_and_preprocess_image(crop)
        predictions = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
    
    # Display results for this crop
    with st.container():
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🚦 {class_labels[predicted_class]['name']}</h3>
            <p>The image shows a <strong>{class_labels[predicted_class]['name'].lower()}</strong> sign, 
            which means <em>{class_labels[predicted_class]['description']}</em>.</p>
            <p><strong>Confidence Level:</strong> <span style="color: #2563eb; font-weight: 700; font-size: 1.1em;">{confidence:.2f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top predictions for this crop
    st.subheader("📊 Detailed Prediction Breakdown")
    top_n = 5
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    
    for i, idx in enumerate(top_indices):
        prob = predictions[0][idx] * 100
        st.write(f"**{i+1}. {class_labels[idx]['name']}**")
        st.progress(float(predictions[0][idx]))
        st.write(f"📈 {prob:.2f}% confidence")
        st.write(f"💡 *{class_labels[idx]['description']}*")
        if i < len(top_indices) - 1:
            st.write("---")

# App title and description
st.title("🚦 Advanced Traffic Sign Recognition")
st.markdown("""
<div class="main-container">
    <h3>Welcome to our AI-Powered Traffic Sign Recognition System</h3>
    <p>This intelligent system can identify traffic signs from cropped images with high accuracy and low inference time in robust conditions. </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for crops
if 'crops' not in st.session_state:
    st.session_state.crops = []

# File uploader with enhanced options
uploaded_file = st.file_uploader(
    "Drag & drop or select an image containing traffic signs", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Maximum filename length: 100 characters."
)

if uploaded_file is not None:
    # Validate filename length
    if len(uploaded_file.name) > 100:
        st.error("Filename too long. Please use a filename with 100 characters or less.")
    else:
        # Display original image with crop functionality
        original_image = Image.open(uploaded_file)
        
        st.subheader("📸 Original Image")
        st.markdown('<div class="original-image">', unsafe_allow_html=True)
        st.image(original_image, caption=f"Original Dimensions: {original_image.size[0]}×{original_image.size[1]} pixels", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("✂️ Crop Traffic Signs")
        st.write("Select areas containing traffic signs to analyze them individually")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image for cropping
            st.write("**Select an area to crop:**")
            img_array = np.array(original_image)
            
            # Get crop coordinates
            x1 = st.slider("Left", 0, original_image.width-1, 0, key="x1")
            x2 = st.slider("Right", 0, original_image.width-1, original_image.width-1, key="x2")
            y1 = st.slider("Top", 0, original_image.height-1, 0, key="y1")
            y2 = st.slider("Bottom", 0, original_image.height-1, original_image.height-1, key="y2")
            
            # Ensure valid crop coordinates
            if x1 >= x2:
                x2 = x1 + 1
            if y1 >= y2:
                y2 = y1 + 1
            
            # Create and display crop
            crop = original_image.crop((x1, y1, x2, y2))
            st.image(crop, caption=f"Crop Preview: {crop.size[0]}×{crop.size[1]} pixels", use_container_width=True)
            
            # Button to add crop
            if st.button("Add Crop for Analysis"):
                crop_bytes = BytesIO()
                crop.save(crop_bytes, format='PNG')
                st.session_state.crops.append({
                    'image': crop,
                    'name': f"Crop {len(st.session_state.crops)+1}",
                    'coords': f"({x1},{y1})-({x2},{y2})"
                })
                st.success(f"Added {len(st.session_state.crops)} crop(s) for analysis")
        
        with col2:
            st.write("**📋 Image Details:**")
            st.write(f"- **Format:** {original_image.format}")
            st.write(f"- **Size:** {original_image.size[0]} × {original_image.size[1]} px")
            st.write(f"- **Filename:** {uploaded_file.name}")
            st.write(f"- **Mode:** {original_image.mode}")
            
            st.write("**🖱️ Crop Controls**")
            st.write("1. Adjust the sliders to select an area")
            st.write("2. Click 'Add Crop' to save for analysis")
            st.write("3. Analyze all saved crops below")
            
            if st.button("Clear All Crops"):
                st.session_state.crops = []
                st.success("All crops cleared")
            
            if len(st.session_state.crops) > 0:
                st.write("**📌 Saved Crops**")
                for i, crop_data in enumerate(st.session_state.crops):
                    st.write(f"{i+1}. {crop_data['name']} {crop_data['coords']}")
        
        # Analyze all saved crops
        if len(st.session_state.crops) > 0:
            st.subheader("🔍 Analysis Results for All Crops")
            
            for i, crop_data in enumerate(st.session_state.crops):
                with st.expander(f"### {crop_data['name']} - {crop_data['coords']}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(crop_data['image'], caption=crop_data['name'], use_container_width=True)
                    
                    with col2:
                        analyze_cropped_image(crop_data['image'], crop_data['name'])

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.header("📖 About This App")
    st.markdown("""
    This AI-powered application recognizes 43 different traffic signs using a custom Convolutional Neural Network trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The crop functionality in this application effectively simulates object detection capabilities by allowing users to manually identify and isolate individual traffic signs within an image for analysis. You can experience lightning-fast predictions and detailed explanations of each recognized sign.
    The model achieves high accuracy in robust conditions, making it suitable for real-world applications.
    """)
    
    st.header("🚀 How To Use")
    st.markdown("""
    1. **Upload** an image containing traffic signs
    2. **Crop** individual signs using the sliders
    3. **Add** crops to analyze them (please crop complete traffic sign for better results)
    4. **View** detailed results for each sign
    """)
    
    st.header("🤖 Model Information")
    st.markdown("""
    **Model Details:**
    - **Architecture:** Lightweight custom Convolutional Neural Network (CNN)
    - **Accuracy:** ~98% on the test set
    - **Input Size:** 48 × 48 pixels (RGB)
    - **Classes:** 43 distinct traffic signs
    - **Training Data:** German Traffic Sign Recognition Benchmark (GTSRB)
    - **Framework:** TensorFlow / Keras
    """)

    st.header("🌟 Application Overview")
    st.markdown("""
    This application is designed to integrate traffic sign detection and recognition into various systems, enhancing road safety and navigation:
    
    **🚗 Autonomous Vehicles**: Real-time traffic sign recognition for enhanced navigation and safety.
    
    **🛡️ Driver Assistance**: Providing alerts and information based on recognized traffic signs.
    
    **🏙️ Traffic Management**: Analyzing traffic sign data for urban planning and optimization.
    
    **📚 Educational Tools**: Helping learners understand traffic signs and their meanings.
    
    **📱 Mobile Applications**: Assisting drivers in unfamiliar areas by recognizing local traffic signs.
    """)

    # Footer
    st.markdown("""
    <div class="footer-content"> 
        <ul class="contact-list">
             <h1>Developed by</h1>
            <li><i class="fas fa-user"></i> Abhiram Madam</li>
            <li><i class="fas fa-graduation-cap"></i> B.Tech in DSAI</li>
            <li><i class="fas fa-university"></i> IIT Guwahati</li>
            <a href="https://github.com/Abhiram0703" class="footer-link" title="Github">
            <li><i class="fab fa-github"></i>Github
            </a><li>
            <li><a href="https://www.instagram.com/abhiram_0703/" class="footer-link" title="Instagram">
            <i class="fab fa-instagram"></i>Instagram
            </a><li>
            <li><a href="https://www.linkedin.com/in/abhiram-madam/" class="footer-link" title="LinkedIn">
            <i class="fab fa-linkedin-in"></i>LinkedIn
            </a><li>
        </ul>
    </div>
    """, unsafe_allow_html=True)