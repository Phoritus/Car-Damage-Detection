import streamlit as st
from PIL import Image
import io
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
        color: #1E3A8A;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #1F2937;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of your vehicle for damage detection"
    )
    
    if uploaded_file:
        # Display file info
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
                        # Import model helper inside the button to avoid loading issues
                        from model_helper import predict_from_image
                        
                        # Make prediction directly from PIL image
                        prediction = predict_from_image(image)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Determine damage severity
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
                                st.metric("Damage Type", prediction.split()[1] if len(prediction.split()) > 1 else "Unknown")
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>❌ Analysis Failed</h3>
                            <p><strong>Error:</strong> {str(e)}</p>
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
<div style="text-align: center; color: #374151; padding: 1rem; font-weight: 500;">
    <p>🤖 Powered by AI Deep Learning | 🔒 Your images are processed securely and not stored</p>
</div>
""", unsafe_allow_html=True)