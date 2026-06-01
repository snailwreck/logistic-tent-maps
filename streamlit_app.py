import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(["Logistic Map", "Tent Map"])

with tab1:
    st.title("Logistic Map")
    st.write("x(n+1) = r * x(n)(1 - x(n))")

    # --- SIDEBAR CONFIGURATION (Moved to top to prevent NameErrors) ---
    st.sidebar.header("Bifurcation Parameters")
    r_min = st.sidebar.slider("Min r", 2.5, 4.0, 2.5)
    r_max = st.sidebar.slider("Max r", 2.5, 4.0, 4.0)
    n_iterations = st.sidebar.number_input("Iterations per r", value=1000)
    n_discard = st.sidebar.number_input("Discard initial", value=100)

    st.sidebar.markdown("---")
    st.sidebar.header("Time Series Analysis Paramters")
    r_val = st.sidebar.slider("Select r for System Graph & Time Series", 2.5, 4.0, 3.5)
    n_time_steps = st.sidebar.number_input("Time steps", value=100)


    # --- 1. DYNAMICAL SYSTEM GRAPH (Cobweb Plot) ---
    st.header(f"System Graph & Cobweb Plot (r = {r_val})")
    st.write("Visualizes the mapping function against the identity line y = x to show state transitions.")

    def generate_cobweb_plot(r, steps):
        fig, ax = plt.subplots(figsize=(6, 6))
        x_vals = np.linspace(0, 1, 500)
        
        # Plot the system function and equilibrium line
        ax.plot(x_vals, r * x_vals * (1 - x_vals), 'r', label="f(x) = r*x(1-x)")
        ax.plot(x_vals, x_vals, 'k--', label="y = x")
        
        # Generate and plot the trajectory path (cobweb lines)
        x_current = 0.5
        # Limit visual steps to 50 max so the plot remains readable
        visual_steps = min(steps, 50) 
        
        for _ in range(visual_steps):
            y_next = r * x_current * (1 - x_current)
            # Vertical line to the curve
            ax.plot([x_current, x_current], [x_current, y_next], 'g', alpha=0.5, lw=1)
            # Horizontal line to the y=x line
            ax.plot([x_current, y_next], [y_next, y_next], 'g', alpha=0.5, lw=1)
            x_current = y_next
            
        ax.set_xlabel("x_n")
        ax.set_ylabel("x_(n+1)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.2)
        return fig

    fig_cobweb = generate_cobweb_plot(r_val, n_time_steps)
    st.pyplot(fig_cobweb)


    # --- 2. BIFURCATION DIAGRAM ---
    st.markdown("---")
    st.header("Bifurcation Diagram")

    def generate_bifurcation_data(r_min, r_max, n_iterations, n_discard):
        r_values = np.linspace(r_min, r_max, 1000)
        x = 0.5 * np.ones(len(r_values))
        
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


    # --- 3. TIME SERIES ANALYSIS ---
    st.markdown("---")
    st.header(f"Time Series Analysis (r = {r_val})")

    def generate_time_series(r, steps):
        x = 0.5
        series = []
        for _ in range(steps):
            x = r * x * (1 - x)
            series.append(x)
        return series

    if st.button("Show Time Series"):
        data = generate_time_series(r_val, n_time_steps)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data, marker='o', linestyle='-', markersize=4, color='b')
        ax.set_title(f"Time Series for r = {r_val}")
        ax.set_xlabel("n")
        ax.set_ylabel("x_n")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)


    # --- 4. FEIGENBAUM CONSTANT ---
    st.markdown("---")
    st.header("Feigenbaum Constant (delta)")
    st.write("Calculates the ratio of successive period-doubling bifurcation intervals.")

    if st.button("Calculate Feigenbaum Constant"):
        with st.spinner("Calculating superstable bifurcation parameters..."):
            
            def get_bifurcation_parameter(n_periods, initial_guess):
                r = initial_guess
                for _ in range(5):
                    x = 0.5
                    dx_dr = 0
                    for _ in range(2**n_periods):
                        dx_dr = x * (1 - x) + r * (1 - 2 * x) * dx_dr
                        x = r * x * (1 - x)
                    r -= (x - 0.5) / dx_dr
                return r

            mus = [3.0, 3.44948974, 3.54409036] 
            for i in range(3, 7):
                guess = mus[-1] + (mus[-1] - mus[-2]) / 4.669
                mus.append(get_bifurcation_parameter(i, guess))

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Bifurcation Parameters (mu_n):**")
                for i, mu in enumerate(mus):
                    st.write(f"Period {2**i}: **{mu:.6f}**")
                    
            with col2:
                st.markdown("**Delta Approximation:**")
                for i in range(2, len(mus) - 1):
                    delta = (mus[i] - mus[i-1]) / (mus[i+1] - mus[i])
                    st.write(f"n={i}: **{delta:.6f}**")