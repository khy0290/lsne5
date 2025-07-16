import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pandas as pd

# --- Page Configuration and Theming ---
st.set_page_config(
    page_title="Gravitational Lensing Simulator",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a universe theme
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #0E002A;
        color: #FFFFFF;
    }
    /* Sidebar background */
    .st-emotion-cache-16txtl3 {
        background-color: #1C004F;
    }
    /* Text color */
    h1, h2, h3, h4, h5, h6, p, .st-emotion-cache-10trblm {
        color: #FFFFFF;
    }
    /* Slider labels */
    .st-emotion-cache-ue6h4q {
        color: #E0D6FF !important;
    }
    /* Button styling */
    .stButton>button {
        color: #FFFFFF;
        background-color: #6C42D4;
        border: 1px solid #8A63D4;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8A63D4;
        border-color: #6C42D4;
    }
    /* Metric labels */
    .st-emotion-cache-1g8m9in {
        color: #C9B8FF;
    }
</style>
""", unsafe_allow_html=True)


# --- Physics Constants & Simulation Parameters (scaled for visualization) ---
# These are not real-world values but are scaled for a clear visual simulation.
G = 6.674e-11  # Not used directly, mass is relative
C = 299792458   # Not used directly, light path is illustrative
SIM_WIDTH = 400 # Width of the simulation view
SIM_HEIGHT = 300 # Increased height for better visual separation
TIME_STEPS = 400 # Number of steps in the simulation


# --- Session State Initialization ---
# This ensures our variables persist between user interactions.
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'time_step' not in st.session_state:
    st.session_state.time_step = 0
if 'light_curve_data' not in st.session_state:
    st.session_state.light_curve_data = pd.DataFrame(columns=['Time', 'Magnification'])
if 'lens_pos_x' not in st.session_state:
    st.session_state.lens_pos_x = -SIM_WIDTH / 2


# --- Helper Functions ---
def calculate_magnification(u, source_radius, einstein_radius):
    """
    Calculates magnification considering the finite size of the source star.
    This prevents the magnification from becoming infinite at perfect alignment.
    u: Impact parameter (distance between lens and source line-of-sight).
    source_radius: The visual radius of the source star.
    einstein_radius: The calculated Einstein radius for the lens.
    """
    # Scale the source radius relative to the Einstein radius
    # Add epsilon to prevent division by zero if einstein_radius is 0
    epsilon = 1e-9
    rho = source_radius / (einstein_radius + epsilon)

    if u <= rho:
        # The lens is transiting the source star.
        # We approximate the magnification by using the value at the edge (u=rho)
        # to create a "flat-topped" peak, which is a good physical approximation.
        if rho < epsilon: # If source is effectively a point
             return (u**2 + 2) / ((u + epsilon) * np.sqrt(u**2 + 4))
        u_calc = rho
        magnification = (u_calc**2 + 2) / (u_calc * np.sqrt(u_calc**2 + 4))
    else:
        # Standard point-lens formula for when the lens is outside the source.
        magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    return magnification

def reset_simulation():
    """Resets the simulation to its initial state."""
    st.session_state.is_running = False
    st.session_state.time_step = 0
    st.session_state.light_curve_data = pd.DataFrame(columns=['Time', 'Magnification'])
    st.session_state.lens_pos_x = -SIM_WIDTH / 2


# --- UI Sidebar ---
with st.sidebar:
    st.title("🌌 Lensing Controls")
    st.markdown("Adjust the parameters of the celestial bodies and the simulation.")

    # --- Star 2 (Lens) Controls ---
    st.header("Lens Star (Star 2)")
    lens_mass = st.slider("Mass (Relative)", 1.0, 10.0, 5.0, 0.1, help="Higher mass causes stronger light bending.")
    lens_radius = st.slider("Radius (Visual)", 5, 20, 10, 1, help="Visual size in the simulation.")
    lens_dist_factor = st.slider("Distance from Earth", 0.1, 0.9, 0.5, 0.05, help="Fraction of the distance between Earth and the Source Star.")

    # --- Planet Controls ---
    st.header("Planet (Optional)")
    has_planet = st.checkbox("Star 2 has a planet", value=True)
    if has_planet:
        planet_mass = st.slider("Planet Mass (Relative)", 0.01, 0.5, 0.1, 0.01, help="A tiny fraction of the lens star's mass.")
        planet_radius = st.slider("Planet Radius (Visual)", 1, 5, 2, 1)
        planet_orbit_dist = st.slider("Planet Orbit Distance", 20, 60, 40, 1, help="Distance from its parent star (Star 2).")
        planet_orbit_speed = st.slider("Planet Orbit Speed", 0.5, 5.0, 2.0, 0.1)
    else:
        # Set defaults if no planet
        planet_mass = 0.0
        planet_radius = 0
        planet_orbit_dist = 0
        planet_orbit_speed = 0

    # --- Source Star (Star 1) Controls ---
    st.header("Source Star (Star 1)")
    source_radius = st.slider("Source Radius (Visual)", 5, 15, 8, 1)

    # --- Simulation Controls ---
    st.header("Simulation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start / Resume", use_container_width=True):
            st.session_state.is_running = True
    with col2:
        if st.button("🔄 Reset", on_click=reset_simulation, use_container_width=True):
            # The on_click handles the reset
            pass

    if st.session_state.is_running:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.is_running = False


# --- Main App Layout ---
st.title("Gravitational Lensing Simulation")
st.markdown(
    "Observe how a massive star (the **Lens**) bends the light from a distant star (the **Source**), "
    "magnifying its brightness as seen from Earth. If a planet is present, it can cause a secondary, smaller spike in brightness."
)

# Create two columns for the simulation and the graph
col_sim, col_graph = st.columns([0.6, 0.4], gap="large")

# Create placeholders to update the plots dynamically
with col_sim:
    sim_placeholder = st.empty()
with col_graph:
    graph_placeholder = st.empty()
    metric_placeholder = st.empty()


# --- Main Simulation Loop ---
while st.session_state.is_running:
    # --- Calculate Positions ---
    # Define vertical positions for clarity
    earth_pos_y = 20
    source_pos_y = SIM_HEIGHT - 20
    
    # Lens Star (Star 2) moves from left to right
    st.session_state.lens_pos_x += SIM_WIDTH / TIME_STEPS
    # The lens's vertical position is now correctly interpolated between Earth and the source
    lens_pos_y = earth_pos_y + (source_pos_y - earth_pos_y) * lens_dist_factor

    # Planet orbits the Lens Star
    angle = (st.session_state.time_step * planet_orbit_speed * 0.1) % (2 * np.pi)
    planet_pos_x = st.session_state.lens_pos_x + planet_orbit_dist * np.cos(angle)
    planet_pos_y = lens_pos_y + planet_orbit_dist * np.sin(angle)

    # Source Star (Star 1) is stationary in the background at the top
    source_pos_x = 0

    # --- Calculate Magnification ---
    # The "Einstein Radius" is proportional to the square root of the mass.
    einstein_radius_star = 15 * np.sqrt(lens_mass)
    impact_parameter_star = abs(st.session_state.lens_pos_x - source_pos_x)
    u_star = impact_parameter_star / einstein_radius_star
    magnification_star = calculate_magnification(u_star, source_radius, einstein_radius_star)

    magnification_planet = 0
    if has_planet:
        einstein_radius_planet = 15 * np.sqrt(planet_mass)
        impact_parameter_planet = abs(planet_pos_x - source_pos_x)
        u_planet = impact_parameter_planet / einstein_radius_planet
        # Planet's effect is additive in this simplified model
        magnification_planet = calculate_magnification(u_planet, source_radius, einstein_radius_planet) - 1.0


    total_magnification = magnification_star + magnification_planet
    
    # Append new data for the light curve
    new_data = pd.DataFrame([[st.session_state.time_step, total_magnification]], columns=['Time', 'Magnification'])
    st.session_state.light_curve_data = pd.concat([st.session_state.light_curve_data, new_data], ignore_index=True)


    # --- Draw Simulation Plot ---
    fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
    ax_sim.set_facecolor('#00001a') # Dark blue space background
    ax_sim.set_xlim(-SIM_WIDTH / 2, SIM_WIDTH / 2)
    ax_sim.set_ylim(0, SIM_HEIGHT)
    ax_sim.set_aspect('equal', adjustable='box')
    ax_sim.set_xticks([])
    ax_sim.set_yticks([])

    # Draw Earth (Observer) at the bottom
    earth = patches.Circle((0, earth_pos_y), radius=5, color='#4da6ff', label='Earth')
    ax_sim.add_patch(earth)
    ax_sim.text(0, earth_pos_y - 15, 'Earth (Observer)', color='white', ha='center', fontsize=10)

    # --- Draw Source Star with Visual Brightness ---
    # Normalize magnification to a 0-1 range for the colormap
    # Capping at 15x magnification for a good color range
    norm_mag = min(1.0, (total_magnification - 1) / 14.0)
    # Use a 'hot' colormap, but scale it to stay in the yellow-orange-white range
    star_color = plt.cm.hot(norm_mag * 0.6 + 0.2)
    source_star = patches.Circle((source_pos_x, source_pos_y), radius=source_radius, color=star_color, label='Source Star')
    ax_sim.add_patch(source_star)
    ax_sim.text(source_pos_x, source_pos_y + source_radius + 5, 'Source Star 1', color='white', ha='center', fontsize=10)


    # Draw Lens Star (Star 2)
    lens_star = patches.Circle((st.session_state.lens_pos_x, lens_pos_y), radius=lens_radius, color='#ff6666', label='Lens Star')
    ax_sim.add_patch(lens_star)

    # Draw Planet
    if has_planet:
        planet = patches.Circle((planet_pos_x, planet_pos_y), radius=planet_radius, color='#99ccff')
        ax_sim.add_patch(planet)

    # --- Draw Light Path (Illustrative) ---
    # Calculate deflection based on mass
    deflection_strength = lens_mass * 0.5
    
    # Path 1 (Top) - Light bends around the top of the lens
    path1_mid_x = st.session_state.lens_pos_x
    path1_mid_y = lens_pos_y + lens_radius + deflection_strength
    ax_sim.plot([source_pos_x, path1_mid_x], [source_pos_y, path1_mid_y], color='yellow', linestyle='--', alpha=0.7)
    ax_sim.plot([path1_mid_x, 0], [path1_mid_y, earth_pos_y], color='yellow', linestyle='--', alpha=0.7,
                label='Bent Light Path')

    # Path 2 (Bottom) - Light bends around the bottom of the lens
    path2_mid_x = st.session_state.lens_pos_x
    path2_mid_y = lens_pos_y - lens_radius - deflection_strength
    ax_sim.plot([source_pos_x, path2_mid_x], [source_pos_y, path2_mid_y], color='yellow', linestyle='--', alpha=0.7)
    ax_sim.plot([path2_mid_x, 0], [path2_mid_y, earth_pos_y], color='yellow', linestyle='--', alpha=0.7)
    
    # Add an arrow to one path to show direction
    ax_sim.annotate('', xy=(0, earth_pos_y), xytext=(path1_mid_x, path1_mid_y),
                arrowprops=dict(arrowstyle="->", color='yellow', lw=1.5))

    sim_placeholder.pyplot(fig_sim)
    plt.close(fig_sim) # Close figure to free memory


    # --- Draw Light Curve Graph ---
    fig_graph, ax_graph = plt.subplots(figsize=(6, 4))
    ax_graph.set_facecolor('#1a0033')
    ax_graph.set_title("Apparent Brightness of Source Star", color='white')
    ax_graph.set_xlabel("Time (steps)", color='white')
    ax_graph.set_ylabel("Magnification (Brightness)", color='white')
    ax_graph.tick_params(colors='white')
    ax_graph.spines['bottom'].set_color('white')
    ax_graph.spines['top'].set_color('white')
    ax_graph.spines['left'].set_color('white')
    ax_graph.spines['right'].set_color('white')
    
    if not st.session_state.light_curve_data.empty:
        ax_graph.plot(st.session_state.light_curve_data['Time'], st.session_state.light_curve_data['Magnification'], color='#8A63D4', marker='o', markersize=2, linestyle='-')
        ax_graph.set_xlim(0, TIME_STEPS)
        # Dynamically adjust y-axis limit
        max_mag = st.session_state.light_curve_data['Magnification'].max()
        ax_graph.set_ylim(0.9, max(3.0, max_mag * 1.1)) # Minimum ylim of 3.0
    
    graph_placeholder.pyplot(fig_graph)
    plt.close(fig_graph)

    # --- Update Metrics ---
    with metric_placeholder.container():
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Current Time Step", f"{st.session_state.time_step}/{TIME_STEPS}")
        m_col2.metric("Current Magnification", f"{total_magnification:.2f}x")

    # --- Loop Control ---
    st.session_state.time_step += 1
    if st.session_state.lens_pos_x > SIM_WIDTH / 2:
        st.session_state.is_running = False
        st.toast("Simulation Complete!", icon="🎉")

    # Control animation speed
    time.sleep(0.05)

# Final state when not running
if not st.session_state.is_running:
    if st.session_state.time_step > 0:
        st.info("Simulation paused or finished. Press 'Start / Resume' to continue or 'Reset' to start over.")
    else:
        st.info("Adjust the parameters in the sidebar and press 'Start' to begin the simulation.")
