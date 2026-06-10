import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

tab1, tab2, tab3 = st.tabs(["Logistic Map", "Tent Map", "Asymmetric Logistic Map"])

with tab1:
    st.title("Logistic Map")
    st.write("x(n+1) = r * x(n)(1 - x(n))")

    st.sidebar.header("Logistic Map: Graph Parameters") 
    r_gph = st.sidebar.slider("Select r for Graph", 2.5, 4.0, 3.5)
    l_gph_iterations = st.sidebar.number_input("Iterations", value=50, min_value=1, max_value=500)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Logistic Map: Bifurcation Parameters")
    r_min = st.sidebar.slider("Min r", 2.5, 4.0, 2.5)
    r_max = st.sidebar.slider("Max r", 2.5, 4.0, 4.0)
    n_iterations = st.sidebar.number_input("Iterations per r", value=1000)
    n_discard = st.sidebar.number_input("Discard initial", value=100)

    st.sidebar.markdown("---")
    st.sidebar.header("Logistic Map: Time Series Analysis Paramters")
    r_val = st.sidebar.slider("Select r for Time Series", 2.5, 4.0, 3.5)
    n_time_steps = st.sidebar.number_input("Time steps", value=100)

    st.header(f"Dynamical System Plot (r = {r_gph})")
    
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

    if st.button("Generate Logistic Graph"):
        fig_cobweb = generate_cobweb_plot(r_gph, l_gph_iterations)
        st.pyplot(fig_cobweb)

    st.markdown("---")
    st.header(f"Time Series Analysis (r = {r_val})")

    def generate_time_series(r, steps):
        x = 0.5
        series = []
        for _ in range(steps):
            x = r * x * (1 - x)
            series.append(x)
        return series

    if st.button("Show Logistic Time Series"):
        data = generate_time_series(r_val, n_time_steps)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data, marker='o', linestyle='-', markersize=4, color='b')
        ax.set_title(f"Time Series for r = {r_val}")
        ax.set_xlabel("n")
        ax.set_ylabel("x_n")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

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

    if st.button("Generate Logistic Diagram"):
        with st.spinner("Calculating..."):
            r_vals, x_vals = generate_bifurcation_data(r_min, r_max, n_iterations, n_discard)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(r_vals, x_vals, ',k', alpha=0.1)
            ax.set_xlabel("r")
            ax.set_ylabel("x")
            ax.set_title("Bifurcation Diagram of the Logistic Map")
            
            st.pyplot(fig)

    st.markdown("---")
    st.header("Feigenbaum Constant")

    if st.button("Calculate Feigenbaum Constant"):
        with st.spinner("Calculating superstable parameters..."):
            
            def get_superstable_r(n_periods, initial_guess, max_iter=30, tol=1e-12):
                """Uses Newton's method to find the exact superstable r parameter for period 2^n."""
                r = initial_guess
                for _ in range(max_iter):
                    x = 0.5
                    dx_dr = 0.0
                    
                    # Track the orbit sequence and its derivative relative to r
                    for _ in range(2**n_periods):
                        dx_dr = x * (1 - x) + r * (1 - 2 * x) * dx_dr
                        x = r * x * (1 - x)
                    
                    if abs(dx_dr) < 1e-12:
                        break
                    step = (x - 0.5) / dx_dr
                    r -= step
                    if abs(step) < tol:
                        break
                return r

            # Seed the array with the true first two superstable points:
            # Period 2^0 (1): r = 2.0
            # Period 2^1 (2): r = 1 + sqrt(5) ~ 3.236068
            r_stable = [2.0, 1.0 + np.sqrt(5)]
            
            # Dynamically discover the higher period superstable values up to period 32 (2^5)
            # Using precise localized guesses to guide the solver safely
            guesses = [3.49, 3.55, 3.565, 3.569]
            for i in range(2, 6):
                r_val = get_superstable_r(i, guesses[i-2])
                r_stable.append(r_val)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Period Doublings:**")
                for i, r_val in enumerate(r_stable):
                    st.write(f"Period {2**i}: **{r_val:.6f}**")
                    
            with col2:
                st.markdown("**Delta Approximation (delta):**")
                for i in range(1, len(r_stable) - 1):
                    numerator = r_stable[i] - r_stable[i-1]
                    denominator = r_stable[i+1] - r_stable[i]
                    
                    if abs(denominator) < 1e-12:
                        st.write(f"n={i}: **N/A (Precision limit)**")
                    else:
                        delta = numerator / denominator
                        st.write(f"n={i}: **{delta:.6f}**")
with tab2:
    st.title("Tent Map")
    st.write("x(n+1) = r * min(x(n), 1 - x(n))")

    # --- SIDEBAR CONFIGURATION (Unique keys added) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Tent Map: Graph Paramters")
    t_r_gph = st.sidebar.slider("Select r for the graph", 0.0, 2.0, 1.5, key="t_r_gph")
    t_gph_iterations = st.sidebar.number_input("Iterations", value=50, min_value=1, max_value=500)

    st.sidebar.markdown("---")
    st.sidebar.header("Tent Map: Bifurcation Paramters")
    t_r_min = st.sidebar.slider("Min r", 0.0, 2.0, 1.0, key="t_r_min")
    t_r_max = st.sidebar.slider("Max r", 0.0, 2.0, 2.0, key="t_r_max")
    t_n_iters = st.sidebar.number_input("Iterations per r", value=1000, key="t_n_iters")
    t_n_discard = st.sidebar.number_input("Discard initial", value=100, key="t_n_discard")

    st.sidebar.markdown("---")
    st.sidebar.header("Tent Map: Time Series Analysis Parameters")
    t_r_val = st.sidebar.slider("Select r for the time series", 0.0, 2.0, 1.5, key="t_r_val")
    t_n_steps = st.sidebar.number_input("Time steps", value=100, key="t_n_steps")

    st.header(f"Dynamical System Plot (r = {t_r_gph})")

    def generate_tent_cobweb(r, steps):
        fig, ax = plt.subplots(figsize=(6, 6))
        x_vals = np.linspace(0, 1, 500)
        
        # Tent map function
        f_x = r * np.minimum(x_vals, 1 - x_vals)
        
        ax.plot(x_vals, f_x, 'r', label="f(x) = r * min(x, 1-x)")
        ax.plot(x_vals, x_vals, 'k--', label="y = x")
        
        x_current = 0.33 # Slightly off-center initial condition
        visual_steps = min(steps, 50) 
        
        for _ in range(visual_steps):
            y_next = r * min(x_current, 1 - x_current)
            ax.plot([x_current, x_current], [x_current, y_next], 'g', alpha=0.5, lw=1)
            ax.plot([x_current, y_next], [y_next, y_next], 'g', alpha=0.5, lw=1)
            x_current = y_next
            
        ax.set_xlabel("x_n")
        ax.set_ylabel("x_(n+1)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.2)
        return fig
    
    if st.button("Generate Tent Graph", key="t_btn_gph"):
        st.pyplot(generate_tent_cobweb(t_r_gph, t_gph_iterations))

    st.markdown("---")
    st.header(f"Time Series Analysis (r = {t_r_val})")

    def generate_tent_time_series(r, steps):
        x = 0.33
        series = []
        for _ in range(steps):
            x = r * min(x, 1 - x)
            series.append(x)
        return series

    if st.button("Show Tent Time Series", key="t_btn_ts"):
        data = generate_tent_time_series(t_r_val, t_n_steps)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data, marker='o', linestyle='-', markersize=4, color='orange')
        ax.set_title(f"Tent Map Time Series for r = {t_r_val}")
        ax.set_xlabel("n")
        ax.set_ylabel("x_n")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    st.markdown("---")
    st.header("Bifurcation Diagram")

    def generate_tent_bifurcation(r_min, r_max, n_iterations, n_discard):
        r_values = np.linspace(r_min, r_max, 1000)
        x = np.random.rand(len(r_values)) # Random init prevents trapping in 0
        
        bifurcation_points = []
        r_points = []
        
        for i in range(n_iterations):
            x = r_values * np.minimum(x, 1 - x)
            if i >= n_discard:
                bifurcation_points.append(x)
                r_points.append(r_values)
                
        return np.array(r_points).flatten(), np.array(bifurcation_points).flatten()

    if st.button("Generate Tent Diagram", key="t_btn_bif"):
        with st.spinner("Calculating..."):
            r_vals, x_vals = generate_tent_bifurcation(t_r_min, t_r_max, t_n_iters, t_n_discard)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(r_vals, x_vals, ',k', alpha=0.1)
            ax.set_xlabel("r")
            ax.set_ylabel("x")
            ax.set_title("Bifurcation Diagram of the Tent Map")
            
            st.pyplot(fig)

with tab3: