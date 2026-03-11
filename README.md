# K-Means Clustering: Step-by-Step & Math Theory

An interactive web application built with Python and Streamlit that visualizes the K-Means clustering algorithm step-by-step. 

Rather than relying on a black-box machine learning library, this project breaks down Lloyd's Algorithm and the underlying mathematics (such as minimizing the Within-Cluster Sum of Squares) to demonstrate exactly how the algorithm learns and converges.

## Author

**Daniel Edgardo Rodríguez Rivera**
* Location: El Salvador
* LinkedIn: [https://www.linkedin.com/in/daniel-rodriguez-sv/](https://www.linkedin.com/in/daniel-rodriguez-sv/)
* GitHub: [https://github.com/danielrodriguezrivera/](https://github.com/danielrodriguezrivera/)

## Features

* **Interactive Step-by-Step Playback:** Watch the centroids update and points get reassigned iteratively.
* **Math Theory Explanations:** View the objective function and centroid update formulas alongside the visual changes at each step.
* **Elbow Method Visualization:** Use a dynamic side-panel chart to evaluate the optimal number of clusters (K) using Inertia.
* **Customizable Datasets:** Adjust the number of data points, actual clusters, and random seeds to see how the algorithm handles different spatial distributions.

## Tech Stack

* **Python:** Core logic and mathematical operations.
* **Streamlit:** Frontend UI and interactive state management.
* **NumPy:** Vectorized distance calculations (Euclidean distance).
* **Matplotlib:** Data and centroid visualization.
* **Scikit-Learn:** Data generation (make_blobs) and baseline model for the Elbow Method.

## How to Run Locally

1. Clone this repository to your local machine.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run app.py`
4. Open your browser to the local URL provided in the terminal (usually http://localhost:8501).