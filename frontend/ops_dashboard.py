"""
Streamlit Operations Dashboard for Airline RAG System
Interactive UI for querying airline operations knowledge base
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Airline Operations Intelligence",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .citation-box {
        background: #f0f2f6;
        padding: 0.5rem;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #22c55e; }
    .confidence-medium { color: #eab308; }
    .confidence-low { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">✈️ Airline Operations Intelligence</p>', unsafe_allow_html=True)
    st.markdown("Enterprise RAG System for SOP Lookup, Maintenance Reasoning & Delay Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Filters
        st.subheader("Filters")
        fleet_filter = st.multiselect(
            "Fleet Type",
            ["B737", "B777", "B787", "A320", "A350"],
            default=[]
        )
        
        airport_filter = st.multiselect(
            "Airport",
            ["ORD", "EWR", "LAX", "SFO", "IAH", "DEN"],
            default=[]
        )
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        streaming = st.checkbox("Enable Streaming", value=True)
        include_figures = st.checkbox("Include Figures", value=True)
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
            st.success("✅ API Connected")
            for check, status in health.get("checks", {}).items():
                st.write(f"{'✓' if status else '✗'} {check.capitalize()}")
        except:
            st.error("❌ API Disconnected")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Query Operations Knowledge Base")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is the pre-flight checklist for B737?",
            height=100
        )
        
        # Example queries
        with st.expander("📝 Example Queries"):
            examples = [
                "What is the pre-flight inspection procedure for B787?",
                "Why might flight UA234 be delayed due to maintenance?",
                "What are the MEL requirements for APU failure on A320?",
                "What is the crew rest requirement for international flights?"
            ]
            for ex in examples:
                if st.button(ex, key=ex):
                    query = ex
        
        # Submit button
        if st.button("🚀 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing query..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/query/",
                            json={
                                "query": query,
                                "fleet_filter": fleet_filter if fleet_filter else None,
                                "airport_filter": airport_filter if airport_filter else None,
                                "include_figures": include_figures,
                                "stream": False
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display answer
                            st.subheader("📋 Answer")
                            st.markdown(result["answer"])
                            
                            # Display citations
                            if result.get("citations"):
                                st.subheader("📚 Sources")
                                for citation in result["citations"]:
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <strong>{citation['citation_id']}</strong> {citation['source_file']}
                                        {f", Page {citation['page_number']}" if citation.get('page_number') else ""}
                                        <br><small>Relevance: {citation['relevance_score']:.2f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Store in session for metrics display
                            st.session_state['last_result'] = result
                        else:
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Ensure the backend is running.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a query.")
    
    with col2:
        st.subheader("📊 Query Metrics")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Confidence metrics
            conf = result.get("confidence", {})
            overall_conf = conf.get("overall_confidence", 0)
            
            conf_class = "high" if overall_conf > 0.8 else "medium" if overall_conf > 0.6 else "low"
            
            st.metric("Confidence", f"{overall_conf:.1%}")
            st.metric("Query Type", result.get("query_type", "unknown").replace("_", " ").title())
            
            # Latency
            latency = result.get("latency", {})
            st.metric("Total Latency", f"{latency.get('total_ms', 0):.0f}ms")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Retrieval", f"{latency.get('retrieval_ms', 0):.0f}ms")
            with col_b:
                st.metric("Generation", f"{latency.get('generation_ms', 0):.0f}ms")
            
            # Grounding
            st.divider()
            st.write("**Grounding Metrics**")
            st.progress(conf.get("citation_coverage", 0))
            st.caption(f"Citation Coverage: {conf.get('citation_coverage', 0):.1%}")
            
            if conf.get("should_abstain"):
                st.warning(f"⚠️ Low confidence: {conf.get('abstention_reason', 'Unknown')}")
        else:
            st.info("Submit a query to see metrics")


if __name__ == "__main__":
    main()

