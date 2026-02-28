import streamlit as st
import database as db
db.init_db()
import plotly.graph_objects as go 

st.set_page_config(page_title="AI 3D FloorPlan", layout="wide")


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
# Vastu helper moved up so it's available when tabs run
def get_vastu_score(facing, bhk_type):
    # Vastu Score Logic
    score = 0
    recommendations = []
    
    # 1. Facing Logic
    if facing == "North" or facing == "East":
        score += 50
        recommendations.append("✅ Entrance is in a highly positive direction (Ishanya/North).")
    elif facing == "West":
        score += 30
        recommendations.append("⚖️ West facing is neutral; ensure the kitchen is in the South-East.")
    else:
        score += 10
        recommendations.append("⚠️ South facing requires Vastu pyramids or heavy curtains at the entrance.")
    
    # 2. BHK logic (Simple assumption for demo)
    if bhk_type >= 2:
        score += 40
    else:
        score += 30
        
    return score, recommendations

# : LOGIN / SIGNUP ---
if not st.session_state.logged_in:
    st.sidebar.title("🔐 Access Control")
    action = st.sidebar.selectbox("Select Action", ["Login", "Sign Up"])
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    
    if action == "Sign Up":
        if st.sidebar.button("Create Account"):
            if db.signup_user(user, pwd):
                st.sidebar.success("Account Created! Now Login.")
            else:
                st.sidebar.error("User already exists!")
    else:
        if st.sidebar.button("Login"):
            if db.login_user(user, pwd):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.sidebar.error("Invalid Credentials")
    
    st.warning("Please login to see the project demo.")


# --- MAIN APP CONTENT (Logged In) ---
else:
    st.sidebar.success(f"Welcome!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🏗️ Archvision 3D: Neural engine for layout conversion")
    
    
    tab1, tab2 = st.tabs(["📸 Photo to 3D (Module A)", "📋 Manual Input & Price (Module B)"])
    import logic

    
    with tab1:
        st.header("🔄 2D to 3D Pipeline")
        uploaded_file = st.file_uploader("Upload Floor Plan", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            
            col1, col2 = st.columns(2)
            
            
            import cv2
            import numpy as np
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            with col1:
                st.subheader("Original Plan")
                st.image(img, channels="BGR", use_container_width=True)

            # Step 2: Wall Detection (Canny)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            with col2:
                st.subheader("Detected Walls (Edges)")
                st.image(edges, caption="Finding Wall Boundaries", use_container_width=True)

            st.divider()

            # Step 3: Wall Extraction (Vector Lines)
            st.subheader("Extracted Wall Coordinates")
            # Logic for vector lines (Hough Transform)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            # Drawing lines on a black canvas for visualization
            canvas = np.zeros_like(img)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue lines

            st.image(canvas, caption="AI has converted pixels into vector lines", use_container_width=True)

            st.divider()

            # Step 4: Final 3D Structural View
            st.subheader("Final 3D Structural View")
            if st.button("Generate Final 3D Model"):
                with st.spinner("Extruding walls..."):
                    fig_3d = logic.generate_3d_plot(lines, img.shape)
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.success("3D Model Ready with Dimensions!")

    # Tab 2: Manual Input & Price
    with tab2:
        L = st.number_input("Length", value=30)
        B = st.number_input("Breadth", value=40)
        bhk = st.selectbox("BHK", [1,2,3])
        face = st.selectbox("Facing", ["North", "East", "West", "South"])
        if st.button("Predict Price & Check Vastu"):
            area = L * B
            
            # ML Price Prediction call
            model = logic.train_price_model("house_data.csv")
            predicted_price = logic.predict_price(model, area, bhk)
            
            # Vastu Calculation call
            v_score, v_tips = get_vastu_score(face, bhk)
            
            # RESULTS DISPLAY
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Predicted Market Price", f"₹{predicted_price/100000:.2f} Lakhs")
                st.subheader("Vastu Analysis")
                st.write(f"**Score: {v_score}/100**")
                for tip in v_tips:
                    st.write(tip)
                    
            with col_res2:
                st.subheader("3D Volume Preview")
                # Manual 3D Box for the dimensions provided
                fig = go.Figure(data=[go.Mesh3d(
                    x=[0, L, L, 0, 0, L, L, 0],
                    y=[0, 0, B, B, 0, 0, B, B],
                    z=[0, 0, 0, 0, 10, 10, 10, 10], # 10ft height
                    color='lightgreen', opacity=0.5
                )])
                st.plotly_chart(fig)

def get_vastu_score(facing, bhk_type):
    # Vastu Score Logic
    score = 0
    recommendations = []
    
    # 1. Facing Logic
    if facing == "North" or facing == "East":
        score += 50
        recommendations.append("✅ Entrance is in a highly positive direction (Ishanya/North).")
    elif facing == "West":
        score += 30
        recommendations.append("⚖️ West facing is neutral; ensure the kitchen is in the South-East.")
    else:
        score += 10
        recommendations.append("⚠️ South facing requires Vastu pyramids or heavy curtains at the entrance.")
    
    # 2. BHK logic (Simple assumption for demo)
    if bhk_type >= 2:
        score += 40
    else:
        score += 30
        
    return score, recommendations

