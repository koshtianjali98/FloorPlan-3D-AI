import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- MODULE A: IMAGE TO 3D ---
def process_floorplan(image_file):
   
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    return lines, img.shape

def generate_3d_plot(lines, img_shape):
    fig = go.Figure()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

           
            fig.add_trace(go.Mesh3d(
                x=[x1, x2, x2, x1, x1, x2, x2, x1],
                y=[y1, y2, y2, y1, y1, y2, y2, y1],
                z=[0, 0, 0, 0, 30, 30, 30, 30], # 30 unit height
                color='gray', opacity=0.6
            ))
    fig.update_layout(title="3D Layout Representation")
    return fig

# --- MODULE B: ML PRICE PREDICTION ---
def train_price_model(csv_path):
    try:
        df = pd.read_csv(csv_path)
        X = df[['SquareFeet', 'Bedrooms']]
        y = df['Price']
        model = LinearRegression().fit(X, y)
        return model
    except:
        return None