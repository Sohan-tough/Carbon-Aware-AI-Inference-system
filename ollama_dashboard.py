"""
ollama_dashboard.py — UI Module for Multi-Model Comparison.
Renders the 3-panel dashboard and handles state for concurrent execution.
"""

import streamlit as st
import concurrent.futures
import time
from ollama_service import (
    is_ollama_running, 
    start_ollama, 
    get_installed_models, 
    run_ollama_inference
)

def render_ollama_status():
    """Render the Ollama status badge and auto-start logic."""
    if "ollama_started" not in st.session_state:
        st.session_state.ollama_started = False

    is_running = is_ollama_running()
    
    if not is_running and not st.session_state.ollama_started:
        st.session_state.ollama_started = True # Mark as tried immediately
        with st.status("🚀 Ollama not running. Attempting auto-start...", expanded=False):
            success = start_ollama()
            if success:
                st.success("Ollama started successfully!")
            else:
                st.error("Failed to start Ollama automatically. Please ensure it is installed and running.")

    # Status indicator
    is_running = is_ollama_running() # Check again
    if is_running:
        st.markdown('<span class="badge badge-green">● OLLAMA RUNNING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">○ OLLAMA DISCONNECTED</span>', unsafe_allow_html=True)

def render_dashboard():
    """Main entry point for the Comparison Dashboard."""
    
    # Custom CSS for the Ollama panels (reusing existing theme patterns)
    st.markdown("""
        <style>
        .panel-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        .panel-header {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
        }
        .panel-output {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            flex-grow: 1;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.5;
            color: #f1f5f9;
            white-space: pre-wrap;
            border: 1px solid #1e293b;
        }
        .latency-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🤖 Multi-Model Compare</p>', unsafe_allow_html=True)
    
    # Init session state for panels
    if "ollama_responses" not in st.session_state:
        st.session_state.ollama_responses = [None, None, None]
    if "ollama_loading" not in st.session_state:
        st.session_state.ollama_loading = False

    # ── Top Section: Prompt ───────────────────────────────────────────────────
    col_prompt, col_status = st.columns([3, 1])
    
    with col_prompt:
        user_prompt = st.text_area(
            "Enter your prompt",
            placeholder="Ask something to compare models...",
            height=120,
            label_visibility="collapsed"
        )
    
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        render_ollama_status()
        
        models = get_installed_models()
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.rerun()
            
        run_btn = st.button("🚀 Run Models", use_container_width=True, type="primary", disabled=st.session_state.ollama_loading)
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.ollama_responses = [None, None, None]
            st.rerun()

    # ── Main Section: Panels ──────────────────────────────────────────────────
    cols = st.columns(3)
    
    # Model selection persist
    if "panel_models" not in st.session_state:
        st.session_state.panel_models = [
            models[0] if len(models) > 0 else "n/a",
            models[1] if len(models) > 1 else (models[0] if len(models) > 0 else "n/a"),
            models[2] if len(models) > 2 else (models[0] if len(models) > 0 else "n/a"),
        ]

    for i in range(3):
        with cols[i]:
            st.session_state.panel_models[i] = st.selectbox(
                f"Model {i+1}",
                options=models if models else ["No models found"],
                index=min(i, len(models)-1) if models else 0,
                key=f"model_select_{i}"
            )
            
            # Panel Container
            container = st.container()
            with container:
                resp = st.session_state.ollama_responses[i]
                
                # Header
                st.markdown(f"""
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; margin-bottom:5px;">
                        PANEL {i+1}
                    </div>
                """, unsafe_allow_html=True)
                
                # Output area
                if st.session_state.ollama_loading:
                    st.info("Thinking...")
                elif resp:
                    if resp["status"] == "success":
                        st.markdown(f'<div class="panel-output">{resp["response"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="latency-badge">⏱ {resp["latency"]}s · {resp["model"]}</div>', unsafe_allow_html=True)
                        if st.button(f"📋 Copy", key=f"copy_{i}"):
                            # Streamlit doesn't have a simple copy-to-clipboard, but we can show it
                            st.toast("Response copied to memory (conceptual)!")
                    else:
                        st.error(f"Error: {resp['message']}")
                else:
                    st.markdown('<div class="panel-output" style="color:#334155;">Idle... waiting for prompt.</div>', unsafe_allow_html=True)

    # ── Execution Logic ───────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        elif not models:
            st.error("❌ No models selected or installed.")
        else:
            st.session_state.ollama_loading = True
            st.rerun()

    # Triggered rerun handler
    if st.session_state.ollama_loading:
        selected_models = st.session_state.panel_models
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_panel = {
                executor.submit(run_ollama_inference, selected_models[i], user_prompt): i 
                for i in range(3)
            }
            
            for future in concurrent.futures.as_completed(future_to_panel):
                panel_idx = future_to_panel[future]
                try:
                    data = future.result()
                    st.session_state.ollama_responses[panel_idx] = data
                except Exception as exc:
                    st.session_state.ollama_responses[panel_idx] = {
                        "status": "error", 
                        "message": str(exc),
                        "model": selected_models[panel_idx]
                    }
        
        st.session_state.ollama_loading = False
        st.rerun()
