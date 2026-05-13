"""
VEDA Streamlit Dashboard
Interactive UI for running ML workflows and monitoring jobs
"""
import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime
import os

# API Configuration
API_BASE_URL = os.getenv("VEDA_API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="VEDA - Autonomous ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .status-running {
        color: #ff9800;
        font-weight: bold;
    }
    .status-completed {
        color: #4caf50;
        font-weight: bold;
    }
    .status-failed {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def api_request(endpoint, method="GET", data=None, auth_required=True):
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    if auth_required and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 401:
            st.session_state.logged_in = False
            st.session_state.token = None
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def login(username, password):
    """Login and get JWT token"""
    data = {"username": username, "password": password}
    response = api_request("/auth/login", method="POST", data=data, auth_required=False)
    
    if response and "access_token" in response:
        st.session_state.token = response["access_token"]
        st.session_state.logged_in = True
        return True
    return False

def get_health():
    """Get API health status"""
    return api_request("/health", auth_required=False)

def get_stats():
    """Get system statistics"""
    return api_request("/stats", auth_required=False)

def create_workflow(dataset_path, goal):
    """Submit a new workflow"""
    data = {"dataset_path": dataset_path, "goal": goal}
    return api_request("/workflows", method="POST", data=data)

def get_workflow_status(job_id):
    """Get workflow status"""
    return api_request(f"/workflows/{job_id}", auth_required=False)

def list_workflows(limit=10):
    """List all workflows"""
    return api_request(f"/workflows?limit={limit}", auth_required=False)

# ============================================================================
# LOGIN PAGE
# ============================================================================

def show_login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🤖 VEDA Login</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Sign In")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin123")
        
        if st.button("Login", use_container_width=True):
            with st.spinner("Logging in..."):
                if login(username, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.info("💡 Demo credentials: admin/admin123 or user/user123")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def show_dashboard():
    """Display main dashboard"""
    
    # Header
    st.markdown('<div class="main-header">🤖 VEDA Autonomous ML Platform</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=VEDA", use_container_width=True)
        st.markdown("---")
        
        # Health check
        health = get_health()
        if health:
            st.markdown("### 🏥 System Health")
            status = health.get("status", "unknown")
            if status == "healthy":
                st.success(f"✅ {status.upper()}")
            else:
                st.error(f"❌ {status.upper()}")
            
            components = health.get("components", {})
            for comp, stat in components.items():
                icon = "✅" if stat in ["operational", "connected", "configured"] else "⚠️"
                st.caption(f"{icon} {comp}: {stat}")
        
        st.markdown("---")
        
        # Stats
        stats = get_stats()
        if stats:
            st.markdown("### 📊 Statistics")
            wf = stats.get("workflows", {})
            st.metric("Total Workflows", wf.get("total", 0))
            st.metric("Completed", wf.get("completed", 0))
            st.metric("Success Rate", f"{wf.get('success_rate', 0):.1f}%")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🚀 New Workflow", "📋 Job Monitor", "📊 Analytics"])
    
    # TAB 1: Create Workflow
    with tab1:
        st.markdown("### Create New ML Workflow")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
            
            if uploaded_file:
                # Save uploaded file
                save_path = f"data/{uploaded_file.name}"
                os.makedirs("data", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Show preview
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded: {len(df)} rows × {len(df.columns)} columns")
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Goal input
                goal = st.text_area(
                    "ML Goal (describe what you want to predict)",
                    placeholder="E.g., predict customer churn based on purchase history",
                    height=100
                )
                
                if st.button("🚀 Start Workflow", type="primary", use_container_width=True):
                    if goal.strip():
                        with st.spinner("Submitting workflow..."):
                            result = create_workflow(save_path, goal)
                            if result:
                                st.session_state.current_job_id = result.get("job_id")
                                st.success(f"✅ Workflow submitted! Job ID: {result.get('job_id')}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Failed to submit workflow")
                    else:
                        st.warning("⚠️ Please enter a goal description")
        
        with col2:
            st.markdown("### 📝 Quick Examples")
            if st.button("Churn Prediction", use_container_width=True):
                st.code("predict customer churn based on usage patterns")
            if st.button("Fraud Detection", use_container_width=True):
                st.code("detect fraudulent transactions")
            if st.button("Sales Forecast", use_container_width=True):
                st.code("forecast next quarter sales")
    
    # TAB 2: Job Monitor
    with tab2:
        st.markdown("### 📋 Active Workflows")
        
        # Current job tracking
        if st.session_state.current_job_id:
            st.info(f"🎯 Tracking: {st.session_state.current_job_id}")
            
            if st.button("🔄 Refresh Status"):
                st.rerun()
            
            status = get_workflow_status(st.session_state.current_job_id)
            if status:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Status", status.get("status", "unknown").upper())
                with col2:
                    st.metric("Progress", f"{status.get('progress', 0):.0f}%")
                with col3:
                    step = status.get("current_step") or "Waiting"
                    st.metric("Current Step", step)
                with col4:
                    created = status.get("created_at", "")
                    if created:
                        st.metric("Started", created[:19])
                
                # Progress bar
                progress = status.get("progress", 0) / 100.0
                st.progress(progress)
                
                # Status details
                job_status = status.get("status", "")
                if job_status == "completed":
                    st.success("✅ Workflow completed successfully!")
                    result = status.get("result", {})
                    if result:
                        st.json(result)
                elif job_status == "failed":
                    st.error(f"❌ Workflow failed: {status.get('error', 'Unknown error')}")
                elif job_status == "running":
                    st.info("⏳ Workflow is running...")
                    st.button("Auto-refresh in 5s...", disabled=True)
                    time.sleep(5)
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📜 Recent Workflows")
        
        workflows = list_workflows(10)
        if workflows:
            for wf in workflows:
                with st.expander(f"{wf.get('goal', 'No goal')[:50]}... ({wf.get('status')})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Job ID:** `{wf.get('job_id')}`")
                        st.write(f"**Dataset:** {wf.get('dataset_path')}")
                        st.write(f"**Created:** {wf.get('created_at', '')[:19]}")
                    with col2:
                        st.write(f"**Status:** {wf.get('status')}")
                        st.write(f"**Progress:** {wf.get('progress', 0):.0f}%")
                        if st.button("View Details", key=f"view_{wf.get('job_id')}"):
                            st.session_state.current_job_id = wf.get('job_id')
                            st.rerun()
    
    # TAB 3: Analytics
    with tab3:
        st.markdown("### 📊 System Analytics")
        
        stats = get_stats()
        if stats:
            wf_stats = stats.get("workflows", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Workflows", wf_stats.get("total", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Success Rate", f"{wf_stats.get('success_rate', 0):.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Failed", wf_stats.get("failed", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Workflow breakdown
            st.markdown("### 📈 Workflow Status Breakdown")
            status_data = {
                "Status": ["Completed", "Failed", "Running"],
                "Count": [
                    wf_stats.get("completed", 0),
                    wf_stats.get("failed", 0),
                    wf_stats.get("running", 0)
                ]
            }
            st.bar_chart(pd.DataFrame(status_data).set_index("Status"))

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Check if logged in
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()