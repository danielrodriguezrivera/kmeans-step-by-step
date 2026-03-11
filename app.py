import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans # Added for the Elbow Method
import time

# --- Configuration ---
st.set_page_config(page_title="K-Means Math & Visualization", layout="wide")
st.title("K-Means Clustering: Step-by-Step & Math Theory")

st.markdown("""
Explore the mathematical mechanics of K-Means. Use the sidebar to set up your data, 
read through the theory, and use the playback controls to watch the algorithm converge!
""")

# --- Sidebar: Author Info ---
with st.sidebar:
    st.header("Author")
    st.markdown("Daniel Edgardo Rodríguez Rivera") 
    st.markdown("El Salvador")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/daniel-rodriguez-sv/")
    with col2:
        st.link_button("GitHub", "https://github.com/danielrodriguezrivera/")
    st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Data Parameters")
n_samples = st.sidebar.slider("Number of data points", 100, 1000, 300, step=50)
true_k = st.sidebar.slider("Actual number of clusters", 2, 8, 4)
random_state = st.sidebar.number_input("Random Seed", value=42)

st.sidebar.header("Algorithm Parameters")
k = st.sidebar.slider("K (Number of clusters)", 2, 8, 4)
delay = st.sidebar.slider("Auto-Play Delay (seconds)", 0.1, 2.0, 0.5)

# --- Data Generation ---
@st.cache_data
def get_data(n_samples, true_k, random_state):
    X, _ = make_blobs(n_samples=n_samples, centers=true_k, cluster_std=1.2, random_state=random_state)
    return X

X = get_data(n_samples, true_k, random_state)

# --- Sidebar: Elbow Method Chart ---
st.sidebar.markdown("---")
st.sidebar.header("Evaluate K (Elbow Method)")
st.sidebar.info("Look for the 'elbow' in the curve to find the optimal $K$.")

# 1. Cache ONLY the heavy calculations, not the plot itself
@st.cache_data
def get_elbow_data(X, max_k=10):
    wcss_values = []
    K_range = range(1, max_k + 1)
    for k_val in K_range:
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    return list(K_range), wcss_values

# Fetch the cached data
K_range, wcss_values = get_elbow_data(X)

# 2. Draw the plot fresh every time so the red line updates instantly
elbow_fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(K_range, wcss_values, marker='o', linestyle='-', color='b')
ax.set_title("Inertia vs. Number of Clusters")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia (WCSS)")
ax.set_xticks(K_range)
ax.grid(True, linestyle='--', alpha=0.6)

# Highlight the current K selected by the user
ax.axvline(x=k, color='red', linestyle='--', label=f'Current K={k}')
ax.legend()

elbow_fig.tight_layout()
st.sidebar.pyplot(elbow_fig)

# --- Math Theory Section ---
with st.expander("📖 Read the Mathematical Theory of K-Means", expanded=False):
    st.markdown("""
    ### The Objective
    K-Means aims to partition $N$ data points into $k$ distinct, non-overlapping clusters. It does this by minimizing the **Within-Cluster Sum of Squares (WCSS)**, also known as Inertia. 
    
    The objective function it tries to minimize is:
    $$J = \sum_{j=1}^{k} \sum_{x_i \in S_j} ||x_i - c_j||^2$$
    
    * $k$ is the total number of clusters.
    * $S_j$ is the set of data points assigned to cluster $j$.
    * $c_j$ is the centroid (mean) of cluster $j$.
    * $||x_i - c_j||^2$ is the squared Euclidean distance between a data point $x_i$ and its centroid $c_j$.
    
    ### Lloyd's Algorithm (The Steps)
    Because finding the absolute mathematically perfect minimum is computationally difficult, K-Means uses an iterative heuristic called Lloyd's Algorithm:
    1.  **Initialization:** Choose $k$ random starting centroids.
    2.  **Assignment:** Assign each point to the closest centroid.
    3.  **Update:** Move the centroid to the mathematical mean of all points assigned to it.
    4.  **Repeat** steps 2 and 3 until the centroids stop moving (convergence).
    """)

# --- Core Algorithm Implementation ---
@st.cache_data
def run_kmeans(X, k, max_iters=50, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.permutation(X.shape[0])[:k]
    centroids = X[initial_indices]
    
    history = []
    
    history.append({
        'iteration': 0,
        'centroids': centroids.copy(),
        'labels': np.zeros(X.shape[0]),
        'wcss': None 
    })
    
    for i in range(max_iters):
        # Assignment Step
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Calculate WCSS
        wcss = 0
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                wcss += np.sum((points_in_cluster - centroids[j])**2)
        
        # Update Step
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
                
        history.append({
            'iteration': i + 1,
            'centroids': new_centroids.copy(),
            'labels': labels.copy(),
            'wcss': wcss
        })
        
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
        
    return history

history = run_kmeans(X, k, random_state=random_state)
max_step = len(history) - 1

# --- State Management for Playback ---
if 'step' not in st.session_state:
    st.session_state.step = 0

# SAFETY CHECK: If the new data converges faster than the old data, 
# our saved step might be out of bounds. Cap it at max_step!
if st.session_state.step > max_step:
    st.session_state.step = max_step

def reset_step():
    st.session_state.step = 0

def next_step():
    if st.session_state.step < max_step:
        st.session_state.step += 1

# --- Playback Controls ---
st.subheader("Interactive Execution")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.button("⏭️ Next Step", on_click=next_step, disabled=(st.session_state.step >= max_step))
with col2:
    st.button("🔄 Reset", on_click=reset_step)
with col3:
    auto_play = st.button("▶️ Auto-Play All")

# --- Visualization Logic ---
ui_placeholder = st.empty()

def render_step(step_index):
    current_state = history[step_index]
    is_converged = (step_index == max_step)
    
    centroids = current_state['centroids']
    centroid_values = "\n".join([f"* **c{i+1}**: ({c[0]:.2f}, {c[1]:.2f})" for i, c in enumerate(centroids)])
    
    wcss = current_state['wcss']
    wcss_display = f"**Current Inertia (WCSS):** {wcss:.2f}" if wcss is not None else "**Current Inertia (WCSS):** N/A"

    math_meaning = (
        "---\n"
        "**What is Inertia?**\n"
        "Inertia measures how tightly grouped the clusters are. It is the sum of the squared distances between each point and its assigned centroid. "
        "Notice how this value drops at each step as the algorithm minimizes the objective function:\n"
        r"$J = \sum_{j=1}^{k} \sum_{x_i \in S_j} ||x_i - c_j||^2$"
    )
    
    with ui_placeholder.container():
        text_col, plot_col = st.columns([1, 2])
        
        with text_col:
            if step_index == 0:
                st.info(
                    "**Initialization:** Randomly selected $k$ initial centroids:\n\n"
                    f"{centroid_values}\n\n"
                    f"{wcss_display}\n\n"
                    "No assignments yet."
                )
            elif is_converged:
                st.success(
                    f"**Convergence (Step {step_index}):** Centroids stabilized.\n\n"
                    r"$c_j^{(new)} = c_j^{(old)}$" + "\n\n"
                    "**Final Centroids:**\n"
                    f"{centroid_values}\n\n"
                    f"{wcss_display}\n\n"
                    f"{math_meaning}"
                )
            else:
                st.info(
                    f"**Iteration {step_index}: Assign & Update**\n\n"
                    "1. **Distance:** Assigned points to nearest centroid:\n"
                    r"$\arg\min_j || x_i - c_j ||^2$" + "\n\n"
                    "2. **Mean:** Centroids moved to the new cluster centers:\n"
                    r"$c_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$" + "\n\n"
                    "**Updated Centroid Coordinates:**\n"
                    f"{centroid_values}\n\n"
                    f"{wcss_display}\n\n"
                    f"{math_meaning}"
                )

        with plot_col:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if step_index == 0:
                ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=30)
            else:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=current_state['labels'], cmap='viridis', alpha=0.6, s=30)
                
            ax.scatter(current_state['centroids'][:, 0], current_state['centroids'][:, 1], 
                       c='red', marker='X', s=150, linewidths=2, edgecolor='black', label='Centroids')
            
            ax.set_title(f"Iteration {step_index}")
            ax.set_xticks([]) 
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.5)
            
            fig.tight_layout() 
            st.pyplot(fig)

# --- Auto-Play Loop Execution ---
if auto_play:
    for s in range(st.session_state.step, max_step + 1):
        st.session_state.step = s
        render_step(s)
        time.sleep(delay)
else:
    render_step(st.session_state.step)