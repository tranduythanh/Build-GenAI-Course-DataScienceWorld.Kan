#!/usr/bin/env python3

import streamlit as st
import traceback
import pandas as pd
import os
import time

# Local imports
from const import (
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_COMMUNITY_FOLDER
)

# Import utility functions for querying
from utils import (
    setup_llm,
    setup_graph_store,
    setup_query_engine,
    load_index_data,
    check_neo4j_connection,
    create_triplets_graph,
    create_communities_graph
)


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
    metadata, files_df, nodes = load_index_data()
    
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
    st.sidebar.write(f"**Nodes (chunks):** {metadata.get('nodes_count', 'Unknown')}")
    
    # Community Summary Status
    st.sidebar.header("🏘️ Community Status")

    # Ensure community folder exists
    os.makedirs(DEFAULT_COMMUNITY_FOLDER, exist_ok=True)

    community_file_path = os.path.join(DEFAULT_COMMUNITY_FOLDER, "summary.json")
    
    if os.path.exists(community_file_path):
        st.sidebar.success("✅ Community summaries cached")
        try:
            import json
            with open(community_file_path, 'r') as f:
                data = json.load(f)
                community_count = len(data.get('community_summaries', {}))
                st.sidebar.write(f"**Communities:** {community_count}")
        except:
            st.sidebar.warning("⚠️ Cache file corrupted")
    else:
        st.sidebar.warning("⚠️ No community cache found")
        st.sidebar.info("Communities will be built from Neo4j")
    
    # Navigation Menu
    st.sidebar.header("🧭 Navigation")
    selected_page = st.sidebar.radio(
        "Select function:",
        ["🔍 Query Interface", "📊 Graph Analysis"],
        index=0
    )
    
    # Initialize session state
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'graph_store' not in st.session_state:
        st.session_state.graph_store = None
    if 'nodes' not in st.session_state:
        st.session_state.nodes = None
    if 'index_loaded' not in st.session_state:
        st.session_state.index_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_status' not in st.session_state:
        st.session_state.current_status = None
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
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
                query_engine = setup_query_engine(index, llm, DEFAULT_SIMILARITY_TOP_K, nodes)
                if not query_engine:
                    st.error("Failed to setup query engine")
                    return
                
                # Store in session state
                st.session_state.query_engine = query_engine
                st.session_state.graph_store = graph_store
                st.session_state.index = index
                st.session_state.nodes = nodes
                st.session_state.index_loaded = True
                
                st.success("✅ Query interface ready!")
                
            except Exception as e:
                st.error(f"Error setting up query interface: {e}")
                st.error(traceback.format_exc())
                return
    
    # Main content area - display based on sidebar selection
    if selected_page == "🔍 Query Interface":
        st.header("💬 Query Your Knowledge Graph")
        
        if st.session_state.index_loaded and st.session_state.query_engine:
            # Instructions and examples
            st.markdown("""
            **How to use:** Ask questions about LLM agents, planning, memory systems, and tools. 
            The system will search through the knowledge graph to provide relevant answers.
            
            **Example queries:**
            - *What are the main components of LLM-powered autonomous agents?*
            - *How does planning work in LLM agents?*
            - *What are the different types of memory in agent systems?*
            """)
            
            # Process query logic (moved before display to handle state changes)
            query_processed = False
            
            # Display chat history first
            if st.session_state.chat_history:
                st.divider()
                st.subheader("💬 Chat History")
                
                for i, chat in enumerate(st.session_state.chat_history):
                    if chat['type'] == 'user':
                        st.markdown(f"""
                        <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                        <strong>🙋‍♂️ You ({chat['timestamp']}):</strong><br>
                        {chat['content']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif chat['type'] == 'assistant':
                        st.markdown(f"""
                        <div style='background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                        <strong>🤖 Assistant ({chat['timestamp']}):</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show response content
                        st.markdown(chat['content'])
                        
                        # Add debug info expander for each response
                        if 'response_obj' in chat:
                            with st.expander(f"🔍 Debug Info for answer {i//2 + 1}", expanded=False):
                                response_obj = chat['response_obj']
                                
                                # Show source nodes if available
                                if hasattr(response_obj, 'source_nodes') and response_obj.source_nodes:
                                    st.write("**📚 Source Nodes Used:**")
                                    for j, node in enumerate(response_obj.source_nodes):
                                        with st.expander(f"Source Node {j+1} (Score: {getattr(node, 'score', 'N/A')})"):
                                            st.write(f"**Content:** {node.text[:500]}...")
                                            if hasattr(node, 'metadata'):
                                                st.write(f"**Metadata:** {node.metadata}")
                                
                                # Show query processing details if available
                                if hasattr(response_obj, 'metadata') and response_obj.metadata:
                                    st.write("**⚙️ Query Processing Details:**")
                                    st.json(response_obj.metadata)
                        
                    elif chat['type'] == 'error':
                        st.markdown(f"""
                        <div style='background-color: #ffebee; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #f44336;'>
                        <strong>❌ Error ({chat['timestamp']}):</strong><br>
                        {chat['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("")  # Add spacing between messages
            
            # Query input interface - now at the bottom
            st.divider()
            st.subheader("✍️ Ask a new question")
            
            # Query input interface
            query_input = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main components of LLM agents?",
                key=f"query_input_{st.session_state.input_key}"
            )
            
            # Ask button
            query_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            # Process query
            if query_button and query_input.strip():
                # Add user question to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': query_input,
                    'timestamp': time.strftime("%H:%M:%S")
                })
                
                try:
                    # Show processing status
                    status_placeholder = st.empty()
                    
                    # Show initial processing message
                    with status_placeholder.container():
                        st.info("🔍 **Processing your question...**")
                        st.write("- Searching in knowledge graph...")
                        st.write("- Analyzing entities and relationships...")
                    
                    # Execute the actual query
                    response = st.session_state.query_engine.query(query_input)
                    
                    # Clear status immediately after getting response
                    status_placeholder.empty()
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': str(response),
                        'response_obj': response,
                        'timestamp': time.strftime("%H:%M:%S")
                    })
                    
                    # Clear the input by incrementing the key
                    st.session_state.input_key += 1
                    
                    # Note: Do not try to clear the input as it causes Streamlit error
                    # st.session_state.query_input = ""  # This line causes the error
                    st.rerun()
                    
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"❌ Error processing question: {str(e)}")
                    st.session_state.chat_history.append({
                        'type': 'error',
                        'content': f"Error: {str(e)}",
                        'timestamp': time.strftime("%H:%M:%S")
                    })
            
            elif query_button and not query_input.strip():
                st.warning("⚠️ Please enter a question before clicking Ask.")
                
        else:
            st.warning("⚠️ Query interface not ready. Please check the setup.")
    
    elif selected_page == "📊 Graph Analysis":
        st.header("📊 Graph Analysis")
        
        if st.session_state.index_loaded and st.session_state.graph_store:
            # Get all triplets from the graph store
            try:
                with st.spinner("Loading all triplets from Neo4j..."):
                    triplets = st.session_state.graph_store.get_triplets()
                
                st.success(f"✅ Loaded {len(triplets)} triplets from the knowledge graph")
                
                # Display triplets in a table format
                if triplets:
                    st.subheader("🔺 All Knowledge Graph Triplets")
                    
                    # Create a DataFrame for better display
                    triplet_data = []
                    for i, (entity1, relation, entity2) in enumerate(triplets):
                        triplet_data.append({
                            "ID": i + 1,
                            "Subject": entity1.name if hasattr(entity1, 'name') else str(entity1),
                            "Predicate": relation.label if hasattr(relation, 'label') else str(relation),
                            "Object": entity2.name if hasattr(entity2, 'name') else str(entity2),
                            "Relation Description": relation.properties.get('relationship_description', 'N/A') if hasattr(relation, 'properties') else 'N/A'
                        })
                    
                    triplets_df = pd.DataFrame(triplet_data)
                    
                    # Display the dataframe with search functionality
                    st.dataframe(
                        triplets_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Show some statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Triplets", len(triplets))
                    with col2:
                        unique_entities = set()
                        for entity1, relation, entity2 in triplets:
                            unique_entities.add(entity1.name if hasattr(entity1, 'name') else str(entity1))
                            unique_entities.add(entity2.name if hasattr(entity2, 'name') else str(entity2))
                        st.metric("Unique Entities", len(unique_entities))
                    with col3:
                        unique_relations = set()
                        for entity1, relation, entity2 in triplets:
                            unique_relations.add(relation.label if hasattr(relation, 'label') else str(relation))
                        st.metric("Unique Relations", len(unique_relations))
                    

                    
                    # Graph Visualizations
                    st.divider()
                    st.subheader("🌐 Graph Visualizations")
                    
                    # Triplets Graph Section
                    st.subheader("🔺 Interactive Knowledge Graph")
                    st.write("**Interactive Knowledge Graph Visualization**")
                    
                    # Add controls for triplets graph
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        max_nodes = st.slider(
                            "Max nodes to display", 
                            min_value=10, 
                            max_value=min(100, len(triplets)), 
                            value=min(50, len(triplets)),
                            help="Limit nodes for better performance"
                        )
                    
                    with st.spinner("Creating triplets graph..."):
                        triplets_fig = create_triplets_graph(triplets, max_nodes)
                        if triplets_fig:
                            st.plotly_chart(triplets_fig, use_container_width=True)
                        else:
                            st.error("Could not create triplets graph")
                    
                    # Communities Graph Section
                    st.divider()
                    st.subheader("🏘️ Community Structure")
                    st.write("**Community Structure Visualization**")
                    
                    with st.spinner("Creating communities graph..."):
                        communities_fig = create_communities_graph(st.session_state.graph_store)
                        if communities_fig:
                            st.plotly_chart(communities_fig, use_container_width=True)
                            
                            # Show community details
                            try:
                                communities = st.session_state.graph_store.get_community_summaries()
                                if communities:
                                    st.subheader("📋 Community Details")
                                    for community_id, summary in communities.items():
                                        with st.expander(f"Community {community_id} - Summary"):
                                            st.write(summary)
                            except Exception as e:
                                st.warning(f"Could not load community details: {e}")
                        else:
                            st.warning("Could not create communities graph. Communities may not be built yet.")
                            if st.button("🔨 Build Communities"):
                                with st.spinner("Building communities..."):
                                    try:
                                        # Clear cache file if it exists
                                        os.makedirs(DEFAULT_COMMUNITY_FOLDER, exist_ok=True)
                                        community_file_path = os.path.join(DEFAULT_COMMUNITY_FOLDER, "summary.json")
                                        if os.path.exists(community_file_path):
                                            os.remove(community_file_path)
                                            st.info("Cleared existing community cache")
                                        
                                        # Force rebuild communities
                                        st.session_state.graph_store.community_summary = {}
                                        st.session_state.graph_store.entity_info = None
                                        st.session_state.graph_store.build_communities()
                                        st.success("Communities built successfully! Refresh to see the graph.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error building communities: {e}")
                
                else:
                    st.warning("No triplets found in the knowledge graph.")
                
            except Exception as e:
                st.error(f"Error loading triplets: {e}")
                st.error(traceback.format_exc())
        else:
            st.warning("⚠️ Graph analysis not available. Please check the setup.")
    
    else:
        st.warning("⚠️ Unknown page selected.")
    
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