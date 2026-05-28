import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Logistic Map")

# Sidebar for user configuration
st.sidebar.header("Parameters")
r_min = st.sidebar.slider("Min r", 2.5, 4.0, 2.5)
r_max = st.sidebar.slider("Max r", 2.5, 4.0, 4.0)
n_iterations = st.sidebar.number_input("Iterations per r", value=1000)
n_discard = st.sidebar.number_input("Discard initial", value=100)

def generate_bifurcation_data(r_min, r_max, n_iterations, n_discard):
    r_values = np.linspace(r_min, r_max, 1000)
    x = 0.5 * np.ones(len(r_values))
    
    # Store points
    bifurcation_points = []
    r_points = []
    
    for i in range(n_iterations):
        x = r_values * x * (1 - x)
        if i >= n_discard:
            bifurcation_points.append(x)
            r_points.append(r_values)
            
    return np.array(r_points).flatten(), np.array(bifurcation_points).flatten()

if st.button("Generate Diagram"):
    with st.spinner("Calculating..."):
        r_vals, x_vals = generate_bifurcation_data(r_min, r_max, n_iterations, n_discard)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_vals, x_vals, ',k', alpha=0.1)
        ax.set_xlabel("r")
        ax.set_ylabel("x")
        ax.set_title("Bifurcation Diagram of the Logistic Map")
        
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.header("Time Series Analysis")
r_val = st.sidebar.slider("Select r for Time Series", 2.5, 4.0, 3.5)
n_time_steps = st.sidebar.number_input("Time steps", value=100)

def generate_time_series(r, steps):
    x = 0.5
    series = []
    for _ in range(steps):
        x = r * x * (1 - x)
        series.append(x)
    return series

if st.sidebar.button("Show Time Series"):
    data = generate_time_series(r_val, n_time_steps)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data, marker='o', linestyle='-', markersize=4)
    ax.set_title(f"Time Series for r = {r_val}")
    ax.set_xlabel("n")
    ax.set_ylabel("x_n")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)