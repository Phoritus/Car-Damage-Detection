import streamlit as st
from PIL import Image
from model_helper import predict
import time
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #566573;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #57A8F7;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #57A8F7;
        border-left: 5px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="main-header">🚗 Vehicle Damage Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image of your vehicle to detect damage type and location</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("📋 Detection Categories")
    st.markdown("""
    **Front Damage:**
    - 🔴 Front Breakage
    - 🟠 Front Crushed  
    - 🟢 Front Normal
    
    **Rear Damage:**
    - 🔴 Rear Breakage
    - 🟠 Rear Crushed
    - 🟢 Rear Normal
    """)
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload a clear image of your vehicle
    2. Ensure the damage area is visible
    3. Wait for AI analysis
    4. View the detection results
    """)
    
    st.header("ℹ️ Supported Formats")
    st.markdown("JPG, JPEG, PNG")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📤 Upload Vehicle Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "png", "jpeg"],  # Added more JPEG variants
        help="Upload a clear image of your vehicle for damage detection"
    )
    
    if uploaded_file:
        # Display file info (removed file type)
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.markdown("**📄 File Details:**")
        for key, value in file_details.items():
            st.text(f"{key}: {value}")

with col2:
    if uploaded_file:
        st.subheader("🖼️ Uploaded Image")
        
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Vehicle Image", use_container_width=True)
            
            # Add analyze button
            if st.button("🔍 Analyze Damage", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your vehicle image..."):
                    try:
                        # Save temporarily for model prediction
                        temp_path = "temp_vehicle_image.jpg"
                        image.save(temp_path)
                        
                        # Simulate processing time for better UX
                        time.sleep(1)
                        
                        # Make prediction
                        prediction = predict(temp_path)
                        
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Determine damage severity and color
                        if "Normal" in prediction:
                            result_color = "🟢"
                            severity = "No Damage"
                            advice = "Your vehicle appears to be in good condition!"
                        elif "Breakage" in prediction:
                            result_color = "🔴"
                            severity = "Severe Damage"
                            advice = "Significant damage detected. Professional repair recommended."
                        else:  # Crushed
                            result_color = "🟠"
                            severity = "Moderate Damage"
                            advice = "Moderate damage detected. Consider professional inspection."
                        
                        # Results display
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>{result_color} Detection Results</h3>
                            <p><strong>Damage Type:</strong> {prediction}</p>
                            <p><strong>Severity:</strong> {severity}</p>
                            <p><strong>Recommendation:</strong> {advice}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional information
                        with st.expander("📊 Detailed Analysis"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Damage Location", prediction.split()[0])
                            with col_b:
                                st.metric("Damage Type", prediction.split()[1])
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>❌ Analysis Failed</h3>
                            <p>Error: {str(e)}</p>
                            <p>Please try uploading a different image or contact support.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    else:
        st.markdown("""
        <div class="info-box">
            <h3>👆 Get Started</h3>
            <p>Upload a vehicle image using the file uploader on the left to begin damage detection analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7F8C8D; padding: 1rem;">
    <p>🤖 Powered by AI Deep Learning | 🔒 Your images are processed securely and not stored</p>
</div>
""", unsafe_allow_html=True)