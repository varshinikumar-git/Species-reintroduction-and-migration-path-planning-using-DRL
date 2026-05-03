import streamlit as st
import os
from PIL import Image
import json
import pandas as pd

# === PAGE CONFIG ===
st.set_page_config(
    page_title="DRL Reintroduction Dashboard",
    page_icon="🦋",
    layout="wide"
)

st.title("🦋 DRL for Species Reintroduction & Migration Path Planning")
st.markdown("Interactive visualization of the trained **Deep Reinforcement Learning (DRL)** agent and its ecological reintroduction outcomes.")

# === PATH CONFIG ===
BASE_DIR = r"F:\DRL PROJECT"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SIDEBAR ===
st.sidebar.header("⚙️ Controls")
section = st.sidebar.radio("Select Section", [
    "Overview",
    "Training Progress",
    "Reintroduction Zones",
    "Reintroduction Analysis",
    "Migration Simulation"
])

# === HELPER ===
def show_image(filename, caption=None):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Missing: {filename}")

# === SECTION 1: OVERVIEW ===
if section == "Overview":
    st.subheader("🌍 Project Overview")
    st.markdown("""
    This DRL framework integrates **reinforcement learning** with **habitat restoration and reintroduction planning** to:
    - Identify **optimal reintroduction zones** for species.
    - Learn **migration trajectories** considering environmental factors.
    - Utilize **restoration potential and habitat quality** as reward signals for policy learning.
    """)
    show_image("restoration_zones.png", "Final Restored Habitat Map")

# === SECTION 2: TRAINING PROGRESS ===
elif section == "Training Progress":
    st.subheader("📈 DRL Training Progress & Spatial Context")

    st.markdown("#### 🗺️ Species Range Context")
    show_image("coverage.png", "Madagascar Boundary and Red-fronted Brown Lemur Coverage")

    st.markdown("#### 🧠 Policy Training Curve")
    show_image("training_results.png", "Reward Curve (DRL Training)")

    st.info("Top: The map visualizes habitat overlap between the lemur range and Madagascar boundaries. "
            "Bottom: The DRL reward curve shows the learning progression towards optimal migration and reintroduction policies.")

    
# === SECTION 3: REINTRODUCTION ZONES ===
elif section == "Reintroduction Zones":
    st.subheader("📍 Reintroduction Site Distribution")
    st.markdown("""
    This section displays **candidate reintroduction sites** derived from DRL and habitat suitability analysis.
    These represent potential zones for species relocation after restoration.
    """)
    show_image("Eulemur_rufifrons_candidate_sites (2).png")
    
    
    st.markdown("""
    **Zone Legend:**
    - 🟢 *High Suitability Site*  
    - 🟡 *Moderate Suitability Site*  
    - 🔴 *Low Suitability Site*  
    """)
    st.info("Each site is derived from restoration-based suitability metrics integrated with DRL policy selection.")

# === SECTION 4: REINTRODUCTION ANALYSIS ===
elif section == "Reintroduction Analysis":
    st.subheader("🌿 Reintroduction Habitat & Connectivity Analysis — *Eulemur rufifrons*")

    col1, col2 = st.columns(2)
    with col1:
        show_image("Eulemur_rufifrons_forest_cover (2).png", "Forest Cover Distribution")
        show_image("Eulemur_rufifrons_safety (2).png", "Habitat Safety Index")
    with col2:
        show_image("Eulemur_rufifrons_quality_distribution (2).png", "Habitat Quality Distribution")
        show_image("Eulemur_rufifrons_connectivity (2).png", "Habitat Connectivity Map")

    st.markdown("---")
    st.subheader("📊 Species Statistics & Candidate Summary")

    csv_path = os.path.join(OUTPUT_DIR, "Eulemur_rufifrons_candidate_sites (1).csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)
    else:
        st.info("Candidate site CSV not found — please check the outputs folder.")

    show_image("Eulemur_rufifrons_stats_panel (2).png", "Habitat Suitability Statistics Panel")
# === SECTION 5: MIGRATION SIMULATION ===
elif section == "Migration Simulation":
    st.subheader("🦢 Migration Path Simulation")
    show_image("migration_path.png", "Migration Path Visualization")
    show_image("RL_path.png", "Reinforcement Learning – Learned Path")
    show_image("comparison.png", "Baseline vs DRL Comparison")
st.markdown("---")
st.caption("Developed for Habitat Restoration DRL — Integrating AI with ecological reintroduction planning 🌿")
