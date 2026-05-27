import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Logistic Map Bifurcation Diagram")

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