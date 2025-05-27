#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG Application - Terminal Version
A complete GraphRAG implementation using LlamaIndex and Neo4j
"""

import os
import re
import pandas as pd
from typing import Any

# Local imports
from const import (
    CSV_KG_EXTRACT_TMPL,
    NEO4J_URI, 
    NEO4J_USERNAME, 
    NEO4J_PASSWORD,
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    DATA_URL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_PATHS_PER_CHUNK,
    DEFAULT_SIMILARITY_TOP_K
)
from graph_rag_extractor import GraphRAGExtractor, parse_csv_triplets_fn
from graph_rag_store import GraphRAGStore
from graph_rag_query_engine import GraphRAGQueryEngine

# LlamaIndex imports
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def parse_fn(response_str: str) -> Any:
    """Parse function for extracting entities and relationships from LLM response."""
    # Use the improved CSV parsing function from graph_rag_extractor
    return parse_csv_triplets_fn(response_str)


def load_data(num_samples: int = 50):
    """Load and prepare news articles data."""
    print(f"Loading {num_samples} news articles...")
    news = pd.read_csv(DATA_URL)[:num_samples]
    
    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]
    
    print(f"Loaded {len(documents)} documents")
    return documents


def setup_llm():
    """Setup and configure the language model."""
    print("Setting up OpenAI LLM...")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    llm = OpenAI(model=DEFAULT_MODEL)
    
    # Test the LLM
    print("Testing LLM connection...")
    response = llm.complete("What is the capital of Vietnam?")
    print(f"LLM Test Response: {response}")
    
    return llm


def setup_embedding_model():
    """Setup the embedding model."""
    print("Setting up embedding model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return embed_model


def create_nodes(documents):
    """Create text nodes/chunks from documents."""
    print("Creating text nodes...")
    splitter = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes")
    return nodes


def setup_kg_extractor(llm):
    """Setup the Knowledge Graph extractor."""
    print("Setting up KG extractor...")
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=CSV_KG_EXTRACT_TMPL,
        max_paths_per_chunk=DEFAULT_MAX_PATHS_PER_CHUNK,
        parse_fn=parse_fn,
    )
    return kg_extractor


def setup_graph_store(llm):
    """Setup the Neo4j graph store."""
    print("Setting up Neo4j graph store...")
    graph_store = GraphRAGStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI
    )
    graph_store.set_llm(llm)
    return graph_store


def build_index(nodes, kg_extractor, graph_store, embed_model, num_nodes=10):
    """Build the Property Graph Index."""
    print(f"Building PropertyGraphIndex with {num_nodes} nodes...")
    index = PropertyGraphIndex(
        nodes=nodes[:num_nodes],
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
        embed_model=embed_model
    )
    
    print("Index built successfully!")
    return index


def analyze_graph(index):
    """Analyze the constructed graph."""
    print("\n=== Graph Analysis ===")
    triplets = index.property_graph_store.get_triplets()
    print(f"Total triplets: {len(triplets)}")
    
    if len(triplets) > 0:
        print("\nSample triplet:")
        print(f"Entity 1: {triplets[0][0].name}")
        print(f"Relation: {triplets[0][1].label}")
        print(f"Entity 2: {triplets[0][2].name}")
        
        if len(triplets) > 10:
            print(f"\nTriplet 10 properties:")
            print(f"Entity properties: {triplets[10][0].properties}")
            print(f"Relation properties: {triplets[10][1].properties}")


def build_communities(index):
    """Build communities and generate summaries."""
    print("\n=== Building Communities ===")
    index.property_graph_store.build_communities()
    print("Communities built and summarized!")


def setup_query_engine(index, llm):
    """Setup the GraphRAG query engine."""
    print("Setting up query engine...")
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
    )
    return query_engine


def run_queries(query_engine):
    """Run sample queries."""
    print("\n=== Running Sample Queries ===")
    
    queries = [
        "What are the main news discussed in the document?",
        "What are the main news in energy sector?",
        "What companies or organizations are mentioned?",
        "What are the key events or developments?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        try:
            response = query_engine.query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")
        print()


def main():
    """Main application entry point."""
    print("=" * 60)
    print("GraphRAG Application - Terminal Version")
    print("=" * 60)
    
    try:
        # Load data
        documents = load_data(num_samples=50)
        
        # Setup components
        llm = setup_llm()
        embed_model = setup_embedding_model()
        
        # Create nodes
        nodes = create_nodes(documents)
        
        # Setup extractors and stores
        kg_extractor = setup_kg_extractor(llm)
        graph_store = setup_graph_store(llm)
        
        # Build index
        index = build_index(nodes, kg_extractor, graph_store, embed_model, num_nodes=10)
        
        # Analyze graph
        analyze_graph(index)
        
        # Build communities
        build_communities(index)
        
        # Setup query engine
        query_engine = setup_query_engine(index, llm)
        
        # Run queries
        run_queries(query_engine)
        
        print("\n=== GraphRAG Pipeline Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 