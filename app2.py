import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import streamlit.components.v1 as components
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

# Load model and scaler
try:
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("🚨 Critical Error: Model or scaler files missing. Ensure 'best_fire_detection_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="FireSense AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "brightness": 300.0,
        "bright_t31": 290.0,
        "frp": 15.0,
        "scan": 1.0,
        "track": 1.0,
        "confidence": "nominal"
    }

# Custom CSS for a spectacular, generative AI-inspired UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0d0d0d, #2c003e);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.6);
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    .light-theme .main {
        background: linear-gradient(135deg, #f5f5f5, #e0e0e0);
        color: #1a1a1a;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff3d00, #ffca28);
        color: #ffffff;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.4s ease;
        box-shadow: 0 6px 12px rgba(255, 61, 0, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 20px rgba(255, 61, 0, 0.6);
        background: linear-gradient(90deg, #ffca28, #ff3d00);
    }
    .stSlider, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        border: 2px solid #ff3d00;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .light-theme .stSlider, .light-theme .stSelectbox {
        background-color: rgba(0, 0, 0, 0.05);
        border: 2px solid #ff3d00;
    }
    .stSlider:hover, .stSelectbox:hover {
        border-color: #ffca28;
        box-shadow: 0 6px 12px rgba(255, 202, 40, 0.5);
    }
    .stSlider label, .stSelectbox label {
        font-size: 18px;
        font-weight: bold;
        color: #ffca28;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    .light-theme .stSlider label, .light-theme .stSelectbox label {
        color: #ff3d00;
        text-shadow: none;
    }
    .header {
        font-size: 64px;
        font-weight: 900;
        color: #ff3d00;
        text-align: center;
        text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.8);
        margin-bottom: 15px;
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    .light-theme .header {
        color: #ff3d00;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .subheader {
        font-size: 26px;
        color: #ffca28;
        text-align: center;
        margin-bottom: 40px;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
    }
    .light-theme .subheader {
        color: #d32f2f;
    }
    .result-box {
        background: linear-gradient(135deg, #0288d1, #4fc3f7);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        color: #ffffff;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        margin-top: 30px;
        animation: slideIn 1s ease-in-out;
    }
    .light-theme .result-box {
        background: linear-gradient(135deg, #0288d1, #80d8ff);
        color: #1a1a1a;
    }
    .sidebar .stMarkdown {
        font-size: 16px;
        color: #ffffff;
        background: linear-gradient(135deg, #2c003e, #1a1a1a);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .light-theme .sidebar .stMarkdown {
        background: linear-gradient(135deg, #ffffff, #e0e0e0);
        color: #1a1a1a;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 10px #ff3d00, 0 0 20px #ff3d00; }
        100% { text-shadow: 0 0 20px #ffca28, 0 0 30px #ffca28; }
    }
    @keyframes slideIn {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .input-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        border: 2px solid #ff3d00;
    }
    .light-theme .input-card {
        background: rgba(0, 0, 0, 0.05);
        border: 2px solid #ff3d00;
    }
    .stProgress .st-bo {
        background-color: #ff3d00;
    }
    .heatmap {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .light-theme .heatmap {
        background: rgba(0, 0, 0, 0.05);
    }
    .fire-marker {
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    .confidence-meter {
        font-size: 16px;
        color: #ffca28;
        text-align: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Apply theme
st.markdown(f'<div class="{st.session_state.theme}-theme">', unsafe_allow_html=True)

# Particle.js background
components.html("""
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <div id="particles-js" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
    <script>
        particlesJS("particles-js", {
            particles: {
                number: { value: 150, density: { enable: true, value_area: 800 } },
                color: { value: ["#ff3d00", "#ffca28", "#d32f2f"] },
                shape: { type: "circle" },
                opacity: { value: 0.8, random: true },
                size: { value: 8, random: true },
                line_linked: { enable: false },
                move: { enable: true, speed: 5, direction: "none", random: true }
            },
            interactivity: {
                detect_on: "canvas",
                events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" } }
            }
        });
    </script>
""", height=0)

# Sidebar
with st.sidebar:
    st.header("🔥 Fire Classifier")
    st.markdown("""
    **Fire Classifier **,by Gaurav Singh Pundir, delivers cosmic precision in classifying fire types from MODIS satellite data.

    **Features:**
    - 🌡️ Brightness & T31
    - 🔥 Fire Radiative Power (FRP)
    - 📍 Scan & Track
    - ✅ Confidence Level

    **Fire Types:**
    - 🌿 Vegetation Fire
    - 🏭 Static Land Source
    - 🌊 Offshore Fire
    """)
    st.markdown("---")
    st.markdown("**By Gaurav Singh Pundir** | Built with Streamlit")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = theme.lower()
    # Enhanced SVG fire icon
    st.markdown("""
        <div style="text-align: center;">
            <svg width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Fire Icon">
                <path d="M12 23C16.4183 23 20 19.4183 20 15C20 12.5 18.5 10.5 16.5 9.5C16.5 7.5 14.5 5 12 5C9.5 5 7.5 7.5 7.5 9.5C5.5 10.5 4 12.5 4 15C4 19.4183 7.58172 23 12 23Z" fill="url(#fireGrad)">
                    <animate attributeName="opacity" values="0.6;1;0.6" dur="1.5s" repeatCount="indefinite"/>
                </path>
                <path d="M12 20C15.3137 20 18 17.3137 18 14C18 12 16.5 10 14.5 9C14.5 7 12.5 5 10 5C7.5 5 5.5 7 5.5 9C3.5 10 2 12 2 14C2 17.3137 4.68629 20 8 20" fill="#ffca28" opacity="0.5">
                    <animate attributeName="opacity" values="0.3;0.7;0.3" dur="2s" repeatCount="indefinite"/>
                </path>
                <defs>
                    <radialGradient id="fireGrad" cx="0.5" cy="0.5" r="0.5">
                        <stop offset="0%" stop-color="#ffca28"/>
                        <stop offset="100%" stop-color="#ff3d00"/>
                    </radialGradient>
                </defs>
            </svg>
        </div>
    """, unsafe_allow_html=True)

# Main app container
st.markdown('<div class="main">', unsafe_allow_html=True)

# Hero section with generative AI-inspired orb
st.markdown('<div class="header">Fire Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Unleash Intelligence to Predict Fire Types</div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center;">
        <svg width="180" height="180" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Fire Orb">
            <circle cx="12" cy="12" r="10" fill="url(#orbGrad)">
                <animate attributeName="r" values="10;12;10" dur="2s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
            </circle>
            <circle cx="12" cy="12" r="6" fill="#ffca28" opacity="0.5">
                <animate attributeName="r" values="6;8;6" dur="1.5s" repeatCount="indefinite"/>
            </circle>
            <defs>
                <radialGradient id="orbGrad" cx="0.5" cy="0.5" r="0.5">
                    <stop offset="0%" stop-color="#ffca28"/>
                    <stop offset="100%" stop-color="#ff3d00"/>
                </radialGradient>
            </defs>
        </svg>
    </div>
""", unsafe_allow_html=True)

# Input form
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### Input Satellite Data", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        brightness = st.slider(
            "🌡️ Brightness",
            min_value=0.0, max_value=500.0, value=st.session_state.inputs["brightness"], step=0.1,
            help="Brightness from MODIS (200–500)",
            key="brightness",
            on_change=lambda: st.session_state.inputs.update({"brightness": st.session_state.brightness})
        )
        bright_t31 = st.slider(
            "🌡️ Brightness T31",
            min_value=0.0, max_value=400.0, value=st.session_state.inputs["bright_t31"], step=0.1,
            help="Brightness T31 (200–400)",
            key="bright_t31",
            on_change=lambda: st.session_state.inputs.update({"bright_t31": st.session_state.bright_t31})
        )
        frp = st.slider(
            "🔥 FRP (MW)",
            min_value=0.0, max_value=100.0, value=st.session_state.inputs["frp"], step=0.1,
            help="Fire Radiative Power in megawatts",
            key="frp",
            on_change=lambda: st.session_state.inputs.update({"frp": st.session_state.frp})
        )

    with col2:
        scan = st.slider(
            "📍 Scan",
            min_value=0.0, max_value=5.0, value=st.session_state.inputs["scan"], step=0.1,
            help="Scan value (0.5–5)",
            key="scan",
            on_change=lambda: st.session_state.inputs.update({"scan": st.session_state.scan})
        )
        track = st.slider(
            "📍 Track",
            min_value=0.0, max_value=5.0, value=st.session_state.inputs["track"], step=0.1,
            help="Track value (0.5–5)",
            key="track",
            on_change=lambda: st.session_state.inputs.update({"track": st.session_state.track})
        )
        confidence = st.selectbox(
            "✅ Confidence Level",
            ["low", "nominal", "high"],
            index=["low", "nominal", "high"].index(st.session_state.inputs["confidence"]),
            help="Detection confidence level",
            key="confidence",
            on_change=lambda: st.session_state.inputs.update({"confidence": st.session_state.confidence})
        )

    # Live input summary
    st.markdown("#### Live Input Summary")
    st.write(f"**Brightness**: {brightness:.1f} | **T31**: {bright_t31:.1f} | **FRP**: {frp:.1f}")
    st.write(f"**Scan**: {scan:.1f} | **Track**: {track:.1f} | **Confidence**: {confidence}")

    # Reset button
    if st.button("Reset Inputs", key="reset"):
        st.session_state.inputs = {
            "brightness": 300.0,
            "bright_t31": 290.0,
            "frp": 15.0,
            "scan": 1.0,
            "track": 1.0,
            "confidence": "nominal"
        }
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Real-time input validation
if any([brightness <= 0, bright_t31 <= 0, frp < 0, scan <= 0, track <= 0]):
    st.warning("⚠️ Invalid input! All values must be positive.", icon="🚨")
else:
    # Combine and scale input
    input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    try:
        scaled_input = scaler.transform(input_data)
    except Exception as e:
        st.error(f"🚨 Error scaling input data: {e}")
        st.stop()

    # Predict button
    if st.button("Analyze Fire Type", key="predict"):
        with st.spinner("🔍 Analyzing Satellite Data..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            
            try:
                prediction = model.predict(scaled_input)[0]
                fire_types = {
                    0: "Vegetation Fire 🌿",
                    2: "Static Land Source 🏭",
                    3: "Offshore Fire 🌊"
                }
                result = fire_types.get(prediction, "Unknown ❓")
                
                # Result with animation
                st.markdown(f'<div class="result-box" aria-label="Prediction Result">Fire Type: {result}</div>', unsafe_allow_html=True)
                
                # Confetti celebration
                components.html("""
                    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
                    <script>
                        confetti({
                            particleCount: 300,
                            spread: 140,
                            origin: { y: 0.6 },
                            colors: ['#ff3d00', '#ffca28', '#d32f2f']
                        });
                    </script>
                """, height=0)

                # Live confidence meter
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(scaled_input)[0]
                    st.markdown("### Prediction Confidence Heatmap")
                    prob_data = {fire_types.get(i, f"Class {i}"): prob * 100 for i, prob in enumerate(probabilities) if i in fire_types}
                    df = pd.DataFrame([prob_data])
                    st.bar_chart(df, height=400, use_container_width=True)
                    max_prob = max(prob_data.values())
                    st.markdown(f'<div class="confidence-meter">Confidence: {max_prob:.2f}%</div>', unsafe_allow_html=True)

                # Fire location visualization (SVG)
                st.markdown("### Fire Location (Sample)")
                st.markdown("""
                    <div style="text-align: center;">
                        <svg width="200" height="200" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="fire-marker" aria-label="Fire Location Marker">
                            <circle cx="12" cy="12" r="8" fill="url(#markerGrad)">
                                <animate attributeName="r" values="8;10;8" dur="1.5s" repeatCount="indefinite"/>
                                <animate attributeName="opacity" values="0.6;1;0.6" dur="1.5s" repeatCount="indefinite"/>
                            </circle>
                            <path d="M12 20C15.3137 20 18 17.3137 18 14C18 12 16.5 10 14.5 9C14.5 7 12.5 5 10 5C7.5 5 5.5 7 5.5 9C3.5 10 2 12 2 14C2 17.3137 4.68629 20 8 20" fill="#ffca28" opacity="0.5">
                                <animate attributeName="opacity" values="0.3;0.7;0.3" dur="2s" repeatCount="indefinite"/>
                            </path>
                            <defs>
                                <radialGradient id="markerGrad" cx="0.5" cy="0.5" r="0.5">
                                    <stop offset="0%" stop-color="#ffca28"/>
                                    <stop offset="100%" stop-color="#ff3d00"/>
                                </radialGradient>
                            </defs>
                        </svg>
                    </div>
                """, unsafe_allow_html=True)

                # PDF report
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                c.setFont("Helvetica-Bold", 16)
                c.setFillColor(colors.red)
                c.drawString(1 * inch, 10 * inch, "Fire Classification Report")
                c.setFont("Helvetica", 12)
                c.setFillColor(colors.black)
                c.drawString(1 * inch, 9.5 * inch, f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                c.drawString(1 * inch, 9 * inch, "Inputs:")
                c.drawString(1.2 * inch, 8.8 * inch, f"- Brightness: {brightness:.1f}")
                c.drawString(1.2 * inch, 8.6 * inch, f"- Brightness T31: {bright_t31:.1f}")
                c.drawString(1.2 * inch, 8.4 * inch, f"- FRP: {frp:.1f}")
                c.drawString(1.2 * inch, 8.2 * inch, f"- Scan: {scan:.1f}")
                c.drawString(1.2 * inch, 8.0 * inch, f"- Track: {track:.1f}")
                c.drawString(1.2 * inch, 7.8 * inch, f"- Confidence: {confidence}")
                c.drawString(1 * inch, 7.5 * inch, f"Prediction: {result}")
                if hasattr(model, "predict_proba"):
                    c.drawString(1 * inch, 7.2 * inch, "Probabilities:")
                    y = 7.0
                    for k, v in prob_data.items():
                        c.drawString(1.2 * inch, y * inch, f"- {k}: {v:.2f}%")
                        y -= 0.2
                c.showPage()
                c.save()
                buffer.seek(0)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=buffer,
                    file_name="firesense_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"🚨 Prediction Error: {e}")
            finally:
                progress.empty()

# Close containers
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)