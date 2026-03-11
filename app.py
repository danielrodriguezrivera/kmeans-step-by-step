import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# --- Configuration ---
st.set_page_config(page_title="Montecarlo Project Forecaster", layout="wide")

# --- 1. SIDEBAR ---
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
    
    # --- Uncertainty Buffer Slider ---
    st.subheader("Uncertainty Buffer")
    
    risk_percent = st.slider(
        "Buffer Percentage", 
        min_value=0, 
        max_value=50, 
        value=10, 
        step=5,
        format="%d%%"
    )
    risk_factor = risk_percent / 100.0
    
    st.caption(f"Current Impact: +/- {risk_percent}% spread")
    
    st.info(
        "**What is this?**\n\n"
        "Even the best estimates miss systemic risks (e.g., unexpected vacations, "
        "new tech learning curve, sick days).\n\n"
        "**How it works:**\n"
        "Sliding this UP pushes your Min/Max estimates further apart during "
        "simulation, creating a flatter, more realistic curve (higher uncertainty).\n\n"
        "**Guide to Buffer Level:**\n"
        "• **0-10%:** Normal project friction.\n"
        "• **15-30%:** High risk (New tech, junior team).\n"
        "• **35-50%:** Critical instability.\n\n"
        "**Why is the limit 50%?**\n\n"
        "In project management, if a task requires more than a 50% buffer, "
        "it is considered 'undefined.' Rather than increasing the buffer, "
        "the task should be broken down into smaller, clearer sub-tasks to reduce systemic risk."
    )
    st.markdown("---")

    st.subheader("Technologies Used")
    st.markdown("""
    - [Streamlit](https://docs.streamlit.io/) - UI Framework
    - [Pandas](https://pandas.pydata.org/docs/) - Data Manipulation
    - [NumPy](https://numpy.org/doc/) - Math & Distributions
    - [Plotly](https://plotly.com/python/) - Interactive Charts
    """)

# --- 2. Helper Functions ---

def get_default_data():
    return pd.DataFrame([
        {"ID": "1", "Task": "Planning", "Min": 2.0, "Likely": 3.0, "Max": 5.0, "Dependencies": ""},
        {"ID": "2", "Task": "Design", "Min": 4.0, "Likely": 6.0, "Max": 10.0, "Dependencies": "1"},
        {"ID": "3", "Task": "Frontend Dev", "Min": 5.0, "Likely": 8.0, "Max": 12.0, "Dependencies": "2"},
        {"ID": "4", "Task": "Backend Dev", "Min": 6.0, "Likely": 9.0, "Max": 14.0, "Dependencies": "2"},
        {"ID": "5", "Task": "Integration", "Min": 3.0, "Likely": 5.0, "Max": 8.0, "Dependencies": "3, 4"},
        {"ID": "6", "Task": "Testing", "Min": 2.0, "Likely": 4.0, "Max": 6.0, "Dependencies": "5"},
    ])

def parse_dependencies(dep_str):
    if not isinstance(dep_str, str) or not dep_str.strip() or dep_str.lower() == 'nan':
        return []
    cleaned_deps = []
    for d in dep_str.split(','):
        d = d.strip()
        if d:
            try:
                val = float(d)
                if val.is_integer():
                    cleaned_deps.append(str(int(val)))
                else:
                    cleaned_deps.append(str(val))
            except ValueError:
                cleaned_deps.append(d)
    return cleaned_deps

def calculate_schedule_path(tasks_data):
    schedule = {} 
    def get_times(task_id):
        if task_id in schedule:
            return schedule[task_id]
        if task_id not in tasks_data:
            return 0.0, 0.0
        task = tasks_data[task_id]
        deps = task['deps']
        duration = task['duration']
        if not deps:
            start = 0.0
        else:
            predecessor_finishes = []
            for dep_id in deps:
                _, p_finish = get_times(dep_id)
                predecessor_finishes.append(p_finish)
            start = max(predecessor_finishes) if predecessor_finishes else 0.0
        finish = start + duration
        schedule[task_id] = (start, finish)
        return start, finish

    final_finish_times = []
    for t_id in tasks_data:
        _, finish = get_times(t_id)
        final_finish_times.append(finish)
    return max(final_finish_times) if final_finish_times else 0, schedule

# --- 3. App Layout ---

st.title("Montecarlo Project Forecaster")

if 'df_tasks' not in st.session_state:
    st.session_state.df_tasks = get_default_data()
if 'editor_key' not in st.session_state:
    st.session_state.editor_key = 0
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None

# --- Excel Copy/Paste Feature ---
with st.expander("Paste Data from Excel / CSV", expanded=False):
    example_text = """ID,Task,Min,Likely,Max,Dependencies
1,Analysis,2,3,5,
2,Design,4,6,8,1
3,Development,10,15,20,2
4,Testing,5,8,12,3"""
    paste_data = st.text_area("Paste here", value=example_text, height=150)
    
    if st.button("Load Pasted Data"):
        if paste_data:
            try:
                sep = '\t' if '\t' in paste_data else ','
                first_line = paste_data.split('\n')[0].lower()
                has_header = ('task' in first_line) and (('min' in first_line) or ('likely' in first_line))
                new_df = pd.read_csv(io.StringIO(paste_data), sep=sep, header=0 if has_header else None, dtype={'ID': str, 'Dependencies': str})
                required_cols = ["ID", "Task", "Min", "Likely", "Max", "Dependencies"]
                for i in range(len(required_cols) - len(new_df.columns)):
                    new_df[f"col_{i}"] = ""
                new_df = new_df.iloc[:, :6]
                new_df.columns = required_cols
                new_df["ID"] = new_df["ID"].astype(str).str.replace(r'\.0$', '', regex=True)
                def clean_num(x): return pd.to_numeric(str(x).replace('days','').strip(), errors='coerce')
                for col in ["Min", "Likely", "Max"]: new_df[col] = new_df[col].apply(clean_num).fillna(0.0)
                new_df["Dependencies"] = new_df["Dependencies"].astype(str).replace("nan", "").str.replace(r'\.0', '', regex=True)
                st.session_state.df_tasks = new_df
                st.session_state.editor_key += 1 
                st.session_state.sim_results = None 
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing data: {e}")

# --- Data Editor ---
st.subheader("1. Task List & Estimates")
col_btn1, col_btn2, _ = st.columns([1, 1, 6])
with col_btn1:
    if st.button("Clear Table"):
        st.session_state.df_tasks = pd.DataFrame(columns=["ID", "Task", "Min", "Likely", "Max", "Dependencies"])
        st.session_state.editor_key += 1 
        st.session_state.sim_results = None 
        st.rerun()
with col_btn2:
    if st.button("Load Default Data"):
        st.session_state.df_tasks = get_default_data()
        st.session_state.editor_key += 1 
        st.session_state.sim_results = None 
        st.rerun()

edited_df = st.data_editor(
    st.session_state.df_tasks,
    key=f"editor_{st.session_state.editor_key}", 
    num_rows="dynamic",
    width="stretch", 
    column_config={
        "Dependencies": st.column_config.TextColumn("Dependencies (IDs)"),
        "Min": st.column_config.NumberColumn("Optimistic", min_value=0.0, required=True),
        "Likely": st.column_config.NumberColumn("Most Likely", min_value=0.0, required=True),
        "Max": st.column_config.NumberColumn("Pessimistic", min_value=0.0, required=True)
    }
)

# --- Simulation ---
if st.button("Run Simulation", type="primary"):
    clean_df = edited_df.copy()
    clean_df["ID"] = clean_df["ID"].astype(str).replace("nan", "").str.strip().str.replace(r'\.0$', '', regex=True)
    clean_df = clean_df[clean_df["ID"] != ""]
    if clean_df.empty:
        st.warning("Please add tasks with valid IDs.")
    else:
        task_blueprints = {str(row['ID']): {"name": str(row['Task']), "min": float(row['Min']), "mode": float(row['Likely']), "max": float(row['Max']), "deps": parse_dependencies(str(row['Dependencies']))} for _, row in clean_df.iterrows()}
        simulations = 10000 
        project_durations, sample_runs = [], []
        with st.spinner("Simulating 10,000 scenarios..."):
            for i in range(simulations):
                run_data, run_details = {}, {}
                for t_id, bp in task_blueprints.items():
                    adj_min, adj_max = bp['min'] * (1 - risk_factor), bp['max'] * (1 + risk_factor)
                    vals = sorted([adj_min, bp['mode'], adj_max])
                    sampled_dur = np.random.triangular(vals[0], vals[1], vals[2]) if vals[2] > 0 else 0.0
                    run_data[t_id] = {"duration": sampled_dur, "deps": bp['deps']}
                    if i < 5: run_details[f"{t_id}: {bp['name']}"] = sampled_dur
                proj_finish_time, _ = calculate_schedule_path(run_data)
                project_durations.append(proj_finish_time)
                if i < 5:
                    run_details["TOTAL PROJECT DURATION"] = proj_finish_time
                    sample_runs.append(run_details)
            st.session_state.sim_results = {"durations": project_durations, "samples": sample_runs, "blueprints": task_blueprints, "risk_applied": risk_factor}
            st.rerun()

if st.session_state.sim_results:
    results = np.array(st.session_state.sim_results["durations"])
    p50, p80, p95 = np.percentile(results, 50), np.percentile(results, 80), np.percentile(results, 95)
    st.divider()
    st.subheader("2. Forecast Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Median (P50)", f"{p50:.1f} Days")
    c2.metric("Safe Bet (P80)", f"{p80:.1f} Days")
    c3.metric("High Certainty (P95)", f"{p95:.1f} Days")
    fig_hist = px.histogram(results, nbins=30, color_discrete_sequence=['#636EFA'])
    fig_hist.add_vline(x=p80, line_dash="dash", line_color="orange", annotation_text="P80")
    fig_hist.update_layout(title="Distribution of Completion Dates", xaxis_title="Days", yaxis_title="Frequency")
    st.plotly_chart(fig_hist, width="stretch")

    # Gantt
    st.subheader("3. Typical Schedule (Based on Most Likely Estimates)")
    bp = st.session_state.sim_results["blueprints"]
    _, likely_schedule = calculate_schedule_path({t_id: {"duration": b['mode'], "deps": b['deps']} for t_id, b in bp.items()})
    gantt_df = pd.DataFrame([{"Task": f"{t_id}: {bp[t_id]['name']}", "Start": pd.Timestamp.now().normalize() + pd.Timedelta(days=s), "Finish": pd.Timestamp.now().normalize() + pd.Timedelta(days=f), "ID": t_id} for t_id, (s, f) in likely_schedule.items()])
    gantt_df['ID_Num'] = pd.to_numeric(gantt_df['ID'], errors='coerce')
    gantt_df = gantt_df.sort_values(by="ID_Num")
    fig_gantt = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Task", color="Task", height=max(400, len(gantt_df)*40+100))
    fig_gantt.update_yaxes(categoryorder='array', categoryarray=gantt_df['Task'].tolist(), autorange="reversed")
    st.plotly_chart(fig_gantt, width="stretch")

    # Explainer
    st.divider()
    with st.expander("How it Works: Behind the Scenes of Monte Carlo"):
        st.markdown("### Deep Dive: Monte Carlo & Risk Theory")
        
        st.markdown("""
        **What is a Monte Carlo Simulation?**
        Named after the famous casino in Monaco, this method relies on repeated random sampling to obtain numerical results. In project management, it allows us to account for the "uncertainty" of human estimates by simulating thousands of possible project futures.
        
        **1. The Triangular Distribution (Modeling Uncertainty)**
        We use a **Triangular Distribution** because it only requires three simple values: the Minimum (Optimistic), Most Likely, and Maximum (Pessimistic). Unlike a Normal "Bell Curve," which assumes symmetry, the Triangular distribution can be "skewed" to reflect the reality that projects are more often late than early.
        """)
        
        
        st.markdown("""
        **2. The Critical Path Method (Resolving the Logic)**
        A project isn't just a list of tasks; it's a network. The **Critical Path Method (CPM)** calculates the longest sequence of dependent tasks. In every simulation run, the app:
        - Samples a random duration for *every* task.
        - Checks the dependencies to find the "Earliest Start" for each task.
        - Determines the total duration based on the path that takes the longest time.
        """)
        

        st.markdown("""
        **3. The Uncertainty Buffer (Systemic Risk)**
        The slider adds a "Senior Twist." It represents systemic risks—things that affect the whole team, like a flu season or a major technical hurdle. Mathematically, it expands the spread between your Optimistic and Pessimistic estimates, "flattening" the final probability curve.
        """)
        
        st.markdown("### Simulation in Action (Sample Iterations)")
        st.markdown("Below are the raw sampled durations for 5 out of the 10,000 runs. Watch how the total project duration fluctuates based on the random sampling of individual task ranges.")
        
        raw_sample_df = pd.DataFrame(st.session_state.sim_results["samples"]).T
        raw_sample_df.columns = [f"Run {i+1}" for i in range(len(raw_sample_df.columns))]
        
        # FIXED: Darker background and white font for readability
        def highlight_final_row(s):
            return ['background-color: #262730; color: white; font-weight: bold' if s.name == 'TOTAL PROJECT DURATION' else '' for _ in s]

        st.table(raw_sample_df.style.format("{:.1f} days").apply(highlight_final_row, axis=1))