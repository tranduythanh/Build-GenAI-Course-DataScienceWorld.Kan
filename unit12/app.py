#!/usr/bin/env python3

import streamlit as st
import traceback
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
    
    # Query configuration
    st.sidebar.header("⚙️ Query Settings")
    similarity_top_k = st.sidebar.slider("Similarity top K", 5, 20, DEFAULT_SIMILARITY_TOP_K)
    
    # Initialize session state
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'graph_store' not in st.session_state:
        st.session_state.graph_store = None
    if 'nodes' not in st.session_state:
        st.session_state.nodes = None
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
                query_engine = setup_query_engine(index, llm, similarity_top_k, nodes)
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
    
    # Main content area
    tab1, tab2 = st.tabs(["🔍 Query Interface", "📊 Graph Analysis"])
    
    with tab1:
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
            
            st.divider()
            
            # Simple query interface
            query_input = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main components of LLM agents?",
                key="query_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                query_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            if query_button and query_input.strip():
                # Create a container for real-time updates
                status_container = st.container()
                response_container = st.container()
                
                with status_container:
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                
                try:
                    # Step 1: Initialize
                    status_placeholder.info("🔍 Analyzing your question...")
                    progress_bar.progress(20)
                    
                    # Step 2: Entity extraction
                    status_placeholder.info("🧠 Finding relevant entities in the knowledge graph...")
                    progress_bar.progress(40)
                    
                    # Step 3: Community search
                    status_placeholder.info("🏘️ Searching through communities...")
                    progress_bar.progress(60)
                    
                    # Step 4: Generate response
                    status_placeholder.info("✍️ Generating response...")
                    progress_bar.progress(80)
                    
                    # Execute query
                    response = st.session_state.query_engine.query(query_input)
                    
                    # Step 5: Complete
                    progress_bar.progress(100)
                    status_placeholder.success("✅ Query completed successfully!")
                    
                    # Display response with debug information
                    with response_container:
                        # Create tabs for response and debug info
                        resp_tab1, resp_tab2 = st.tabs(["📝 Final Response", "🔍 Debug Information"])
                        
                        with resp_tab1:
                            st.subheader("📝 Response:")
                            st.write(response)
                            
                            # Add copy button
                            if st.button("📋 Copy Response"):
                                st.success("Response copied to clipboard!")
                        
                        with resp_tab2:
                            st.subheader("🔍 Debug Information")
                            
                            # Show source nodes if available
                            if hasattr(response, 'source_nodes') and response.source_nodes:
                                st.write("**📚 Source Nodes Used:**")
                                for i, node in enumerate(response.source_nodes):
                                    with st.expander(f"Source Node {i+1} (Score: {getattr(node, 'score', 'N/A')})"):
                                        st.write(f"**Content:** {node.text[:500]}...")
                                        if hasattr(node, 'metadata'):
                                            st.write(f"**Metadata:** {node.metadata}")
                            
                            # Show relevant chunks (NEW!)
                            st.write("**📄 Relevant Text Chunks:**")
                            try:
                                if hasattr(st.session_state.query_engine, 'nodes') and st.session_state.query_engine.nodes:
                                    # Get entities for chunk search
                                    entities = []
                                    try:
                                        entities = st.session_state.query_engine.get_entities(query_input, similarity_top_k)
                                    except:
                                        pass
                                    
                                    # Get relevant chunks
                                    relevant_chunks = st.session_state.query_engine.get_relevant_chunks(query_input, entities)
                                    
                                    if relevant_chunks:
                                        st.write(f"Found {len(relevant_chunks)} relevant chunks:")
                                        
                                        chunk_debug_data = []
                                        for i, chunk in enumerate(relevant_chunks[:5]):  # Show top 5
                                            chunk_debug_data.append({
                                                "ID": i + 1,
                                                "Score": chunk['score'],
                                                "Text Preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                                                "Length": len(chunk['text']),
                                                "Source": chunk['metadata'].get('source', 'Unknown') if chunk['metadata'] else 'Unknown'
                                            })
                                        
                                        debug_chunks_df = pd.DataFrame(chunk_debug_data)
                                        st.dataframe(debug_chunks_df, use_container_width=True)
                                        st.write(f"*Showing top {len(chunk_debug_data)} chunks out of {len(relevant_chunks)} total*")
                                    else:
                                        st.write("No relevant chunks found based on keyword and entity matching.")
                                else:
                                    st.write("No nodes available for chunk analysis.")
                                    
                            except Exception as e:
                                st.error(f"Error loading chunks: {e}")
                            
                            # Show relevant triplets
                            st.write("**🔺 Relevant Triplets:**")
                            try:
                                # Get all triplets and filter for relevance
                                all_triplets = st.session_state.graph_store.get_triplets()
                                
                                # Simple keyword matching for relevance
                                query_keywords = query_input.lower().split()
                                relevant_triplets = []
                                
                                for triplet in all_triplets:
                                    entity1, relation, entity2 = triplet
                                    entity1_name = entity1.name if hasattr(entity1, 'name') else str(entity1)
                                    entity2_name = entity2.name if hasattr(entity2, 'name') else str(entity2)
                                    relation_label = relation.label if hasattr(relation, 'label') else str(relation)
                                    
                                    # Check if any query keyword appears in the triplet
                                    triplet_text = f"{entity1_name} {relation_label} {entity2_name}".lower()
                                    if any(keyword in triplet_text for keyword in query_keywords):
                                        relevant_triplets.append(triplet)
                                
                                if relevant_triplets:
                                    triplet_debug_data = []
                                    for i, (entity1, relation, entity2) in enumerate(relevant_triplets[:10]):  # Show top 10
                                        triplet_debug_data.append({
                                            "ID": i + 1,
                                            "Subject": entity1.name if hasattr(entity1, 'name') else str(entity1),
                                            "Predicate": relation.label if hasattr(relation, 'label') else str(relation),
                                            "Object": entity2.name if hasattr(entity2, 'name') else str(entity2),
                                            "Description": relation.properties.get('relationship_description', 'N/A') if hasattr(relation, 'properties') else 'N/A'
                                        })
                                    
                                    debug_triplets_df = pd.DataFrame(triplet_debug_data)
                                    st.dataframe(debug_triplets_df, use_container_width=True)
                                    st.write(f"*Showing {len(relevant_triplets)} relevant triplets out of {len(all_triplets)} total*")
                                else:
                                    st.write("No directly relevant triplets found based on keyword matching.")
                                    
                            except Exception as e:
                                st.error(f"Error loading triplets: {e}")
                            
                            # Show relevant communities
                            st.write("**🏘️ Community Information:**")
                            try:
                                if hasattr(st.session_state.graph_store, 'get_community_summaries'):
                                    communities = st.session_state.graph_store.get_community_summaries()
                                    if communities:
                                        st.write(f"Found {len(communities)} communities in the knowledge graph:")
                                        
                                        # Show community summaries that might be relevant
                                        query_keywords = query_input.lower().split()
                                        relevant_communities = []
                                        
                                        for community_id, summary in communities.items():
                                            summary_lower = summary.lower()
                                            if any(keyword in summary_lower for keyword in query_keywords):
                                                relevant_communities.append((community_id, summary))
                                        
                                        if relevant_communities:
                                            st.write(f"**Relevant Communities ({len(relevant_communities)}):**")
                                            for community_id, summary in relevant_communities[:5]:  # Show top 5
                                                with st.expander(f"Community {community_id}"):
                                                    st.write(summary)
                                        else:
                                            st.write("No communities found with direct keyword matches.")
                                            # Show first few communities as fallback
                                            st.write("**Sample Communities:**")
                                            for i, (community_id, summary) in enumerate(list(communities.items())[:3]):
                                                with st.expander(f"Community {community_id}"):
                                                    st.write(summary[:300] + "..." if len(summary) > 300 else summary)
                                    else:
                                        st.write("No community summaries available.")
                                else:
                                    st.write("Community functionality not available in current graph store.")
                                    
                            except Exception as e:
                                st.error(f"Error loading communities: {e}")
                            
                            # Show query processing details if available
                            if hasattr(response, 'metadata') and response.metadata:
                                st.write("**⚙️ Query Processing Details:**")
                                st.json(response.metadata)
                            
                except Exception as e:
                    status_placeholder.error(f"❌ Error: {str(e)}")
                    st.error(f"Detailed error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            elif query_button and not query_input.strip():
                st.warning("⚠️ Please enter a question before clicking Ask.")
                
        else:
            st.warning("⚠️ Query interface not ready. Please check the setup.")
    
    with tab2:
        st.header("Graph Analysis")
        
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
                    
                    # Create tabs for different visualizations
                    viz_tab1, viz_tab2 = st.tabs(["🔺 Triplets Graph", "🏘️ Communities Graph"])
                    
                    with viz_tab1:
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
                    
                    with viz_tab2:
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
                                            st.session_state.graph_store.build_communities()
                                            st.success("Communities built successfully! Refresh to see the graph.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error building communities: {e}")
                
                else:
                    st.warning("No triplets found in the knowledge graph.")
                
                # Show community information if available (legacy section)
                st.divider()
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
                        
            except Exception as e:
                st.error(f"Error loading triplets: {e}")
                st.error(traceback.format_exc())
        else:
            st.warning("⚠️ Graph analysis not available. Please check the setup.")
    
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