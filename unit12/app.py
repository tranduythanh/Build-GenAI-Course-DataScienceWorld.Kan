#!/usr/bin/env python3

import streamlit as st
import traceback
import os
import pickle
import pandas as pd

# Local imports
from const import (
    DEFAULT_SIMILARITY_TOP_K
)

# Import utility functions for querying
from utils import (
    setup_llm,
    setup_graph_store,
    setup_query_engine,
    analyze_graph
)


def load_index_data(data_dir="./index_data"):
    """Load pre-built index data from disk."""
    try:
        metadata_path = os.path.join(data_dir, 'index_metadata.pkl')
        files_df_path = os.path.join(data_dir, 'files_df.pkl')
        
        if not os.path.exists(metadata_path):
            return None, None
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load files dataframe if exists
        files_df = None
        if os.path.exists(files_df_path):
            files_df = pd.read_pickle(files_df_path)
        
        return metadata, files_df
        
    except Exception as e:
        st.error(f"Error loading index data: {e}")
        return None, None


def check_neo4j_connection(graph_store):
    """Check if Neo4j has data and is accessible."""
    try:
        triplets = graph_store.get_triplets()
        return len(triplets) > 0, len(triplets)
    except Exception as e:
        st.error(f"Neo4j connection error: {e}")
        return False, 0


def main():
    """Main Streamlit application for querying GraphRAG."""
    st.set_page_config(
        page_title="GraphRAG Query Interface",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 GraphRAG Query Interface")
    st.markdown("Query your pre-built knowledge graph using natural language")
    
    # Check if index has been built
    metadata, files_df = load_index_data()
    
    if metadata is None:
        st.error("❌ No pre-built index found!")
        st.markdown("""
        **Please build the index first by running:**
        ```bash
        python build_index.py
        ```
        
        This will:
        1. Convert HTML files to Markdown
        2. Extract knowledge from documents
        3. Build the knowledge graph in Neo4j
        4. Prepare the query engine
        
        After building the index, refresh this page to start querying.
        """)
        return
    
    # Sidebar for configuration and info
    st.sidebar.header("📊 Index Information")
    st.sidebar.write(f"**Built:** {metadata.get('timestamp', 'Unknown')}")
    st.sidebar.write(f"**Files processed:** {metadata.get('files_processed', 'Unknown')}")
    st.sidebar.write(f"**Index type:** {metadata.get('index_type', 'Unknown')}")
    
    # Query configuration
    st.sidebar.header("⚙️ Query Settings")
    similarity_top_k = st.sidebar.slider("Similarity top K", 5, 20, DEFAULT_SIMILARITY_TOP_K)
    
    # Initialize session state
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'graph_store' not in st.session_state:
        st.session_state.graph_store = None
    if 'index_loaded' not in st.session_state:
        st.session_state.index_loaded = False
    
    # Setup components if not already done
    if not st.session_state.index_loaded:
        with st.spinner("🔧 Setting up query components..."):
            try:
                # Setup LLM
                llm = setup_llm()
                if not llm:
                    st.error("Failed to setup LLM")
                    return
                
                # Setup graph store
                graph_store = setup_graph_store(llm)
                if not graph_store:
                    st.error("Failed to setup graph store")
                    return
                
                # Check Neo4j connection and data
                has_data, triplet_count = check_neo4j_connection(graph_store)
                if not has_data:
                    st.error("❌ No data found in Neo4j! Please run `python build_index.py` first.")
                    return
                
                st.success(f"✅ Connected to Neo4j with {triplet_count} triplets")
                
                # Create a mock index for the query engine
                from llama_index.core import PropertyGraphIndex
                index = PropertyGraphIndex.from_existing(
                    property_graph_store=graph_store,
                    embed_kg_nodes=False
                )
                
                # Setup query engine
                query_engine = setup_query_engine(index, llm, similarity_top_k)
                if not query_engine:
                    st.error("Failed to setup query engine")
                    return
                
                # Store in session state
                st.session_state.query_engine = query_engine
                st.session_state.graph_store = graph_store
                st.session_state.index = index
                st.session_state.index_loaded = True
                
                st.success("✅ Query interface ready!")
                
            except Exception as e:
                st.error(f"Error setting up query interface: {e}")
                st.error(traceback.format_exc())
                return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Query Interface", "📊 Graph Analysis", "📄 Data Info"])
    
    with tab1:
        st.header("Ask Questions About Your Knowledge Graph")
        
        if st.session_state.index_loaded and st.session_state.query_engine:
            # Predefined queries
            st.subheader("🎯 Quick Queries")
            predefined_queries = [
                "What are the main components of LLM-powered autonomous agents?",
                "How does planning work in LLM agents?",
                "What are the different types of memory in agent systems?",
                "What tools and techniques are mentioned for agent development?"
            ]
            
            col1, col2 = st.columns(2)
            for i, query in enumerate(predefined_queries):
                with col1 if i % 2 == 0 else col2:
                    if st.button(f"📝 {query}", key=f"predefined_{i}"):
                        with st.spinner("Processing query..."):
                            try:
                                response = st.session_state.query_engine.query(query)
                                st.success("✅ Query completed!")
                                st.write("**Response:**")
                                st.write(response)
                            except Exception as e:
                                st.error(f"Error processing query: {e}")
            
            st.divider()
            
            # Custom query
            st.subheader("💬 Custom Query")
            custom_query = st.text_area(
                "Enter your custom query:",
                placeholder="Ask anything about LLM agents and the technical content...",
                height=100
            )
            
            if st.button("🔍 Run Custom Query", type="primary"):
                if custom_query.strip():
                    with st.spinner("Processing custom query..."):
                        try:
                            response = st.session_state.query_engine.query(custom_query)
                            st.success("✅ Query completed!")
                            st.write("**Response:**")
                            st.write(response)
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
                else:
                    st.warning("Please enter a query.")
        else:
            st.warning("⚠️ Query interface not ready. Please check the setup.")
    
    with tab2:
        st.header("Graph Analysis")
        
        if st.session_state.index_loaded and 'index' in st.session_state:
            # Analyze graph
            triplets = analyze_graph(st.session_state.index)
            
            # Show community information if available
            if hasattr(st.session_state.graph_store, 'get_community_summaries'):
                try:
                    communities = st.session_state.graph_store.get_community_summaries()
                    if communities:
                        st.subheader("🏘️ Community Summaries")
                        for i, (community_id, summary) in enumerate(communities.items()):
                            if i < 5:  # Show first 5 communities
                                with st.expander(f"Community {community_id}"):
                                    st.write(summary)
                except Exception as e:
                    st.warning(f"Could not load community summaries: {e}")
        else:
            st.warning("⚠️ Graph analysis not available. Please check the setup.")
    
    with tab3:
        st.header("Data Information")
        
        # Show files information if available
        if files_df is not None and not files_df.empty:
            st.subheader("📄 Processed Files")
            st.dataframe(files_df, use_container_width=True)
        else:
            st.info("No file information available")
        
        # Show index metadata
        if metadata:
            st.subheader("🔧 Index Metadata")
            st.json(metadata)
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>GraphRAG Query Interface powered by LlamaIndex, Neo4j, and Streamlit</p>
            <p>Build index with: <code>python build_index.py</code></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 