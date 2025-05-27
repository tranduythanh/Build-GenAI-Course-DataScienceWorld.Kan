#!/usr/bin/env python3
"""
Shared utility functions for GraphRAG application.
These functions are used by both app.py and build_index.py.
"""

import os
import streamlit as st

# Local imports
from const import (
    NEO4J_URI, 
    NEO4J_USERNAME, 
    NEO4J_PASSWORD,
    OPENAI_API_KEY,
    DEFAULT_MODEL,
)
from graph_rag_store import GraphRAGStore
from graph_rag_query_engine import GraphRAGQueryEngine

# LlamaIndex imports
from llama_index.llms.openai import OpenAI


def setup_llm():
    """Setup and configure the language model."""
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        llm = OpenAI(model=DEFAULT_MODEL)
        return llm
    except Exception as e:
        print(f"❌ Error setting up LLM: {e}")
        return None


def setup_graph_store(llm):
    """Setup the Neo4j graph store."""
    try:
        graph_store = GraphRAGStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI
        )
        graph_store.set_llm(llm)
        return graph_store
    except Exception as e:
        print(f"❌ Error setting up graph store: {e}")
        return None


def analyze_graph(index):
    """Analyze the constructed graph."""
    try:
        triplets = index.property_graph_store.get_triplets()
        
        if hasattr(st, 'subheader'):  # Only use Streamlit if available
            st.subheader("📊 Graph Analysis")
            st.metric("Total Triplets", len(triplets))
            
            if len(triplets) > 0:
                st.subheader("Sample Triplet")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Entity 1:**")
                    st.write(triplets[0][0].name)
                with col2:
                    st.write("**Relation:**")
                    st.write(triplets[0][1].label)
                with col3:
                    st.write("**Entity 2:**")
                    st.write(triplets[0][2].name)
                
                if len(triplets) > 10:
                    st.subheader("Triplet Properties (Sample)")
                    st.write("**Entity Properties:**", triplets[10][0].properties)
                    st.write("**Relation Properties:**", triplets[10][1].properties)
        else:
            # Terminal mode
            print(f"📊 Graph Analysis:")
            print(f"  - Total Triplets: {len(triplets)}")
            if len(triplets) > 0:
                print(f"  - Sample Triplet: {triplets[0][0].name} -> {triplets[0][1].label} -> {triplets[0][2].name}")
        
        return triplets
    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
        return []


def setup_query_engine(index, llm, similarity_top_k):
    """Setup the GraphRAG query engine."""
    try:
        query_engine = GraphRAGQueryEngine(
            graph_store=index.property_graph_store,
            llm=llm,
            index=index,
            similarity_top_k=similarity_top_k,
        )
        return query_engine
    except Exception as e:
        print(f"❌ Error setting up query engine: {e}")
        return None 