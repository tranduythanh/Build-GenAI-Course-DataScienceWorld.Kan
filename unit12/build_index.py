#!/usr/bin/env python3
"""
Build Index Script for GraphRAG Application
This script builds the knowledge graph index from markdown files and stores it for later querying.
Run this script before using the Streamlit app.
"""

import os
import sys
import traceback
import pickle
from datetime import datetime

# Local imports
from const import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_PATHS_PER_CHUNK,
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_MODEL
)

# Import shared utility functions from utils.py
from utils import (
    setup_llm,
    setup_graph_store,
    setup_query_engine
)

# Import build-specific utility functions from build_utils.py
from build_utils import (
    convert_html_to_markdown,
    load_data,
    setup_embedding_model,
    create_nodes,
    setup_kg_extractor,
    build_index,
    build_communities
)


def print_progress(message, step=None, total_steps=None):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if step and total_steps:
        print(f"[{timestamp}] Step {step}/{total_steps}: {message}")
    else:
        print(f"[{timestamp}] {message}")


def save_index_data(index, query_engine, files_df, output_dir="./index_data"):
    """Save the built index and related data to disk."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save index metadata
        index_metadata = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(files_df) if files_df is not None else 0,
            'index_type': type(index).__name__,
            'query_engine_type': type(query_engine).__name__
        }
        
        with open(os.path.join(output_dir, 'index_metadata.pkl'), 'wb') as f:
            pickle.dump(index_metadata, f)
        
        # Save files dataframe
        if files_df is not None:
            files_df.to_pickle(os.path.join(output_dir, 'files_df.pkl'))
        
        print(f"✅ Index data saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving index data: {e}")
        return False


def main():
    """Main function to build the GraphRAG index."""
    print("🕸️ GraphRAG Index Builder")
    print("=" * 50)
    
    # Configuration
    print("\n📋 Configuration:")
    num_samples = int(os.getenv('NUM_MARKDOWN_FILES', '50'))  # Increased default from 10 to 50
    # Remove node limit - process all nodes to index complete content
    num_nodes = None  # Process all nodes
    chunk_size = int(os.getenv('CHUNK_SIZE', str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', str(DEFAULT_CHUNK_OVERLAP)))
    max_paths_per_chunk = int(os.getenv('MAX_PATHS_PER_CHUNK', str(DEFAULT_MAX_PATHS_PER_CHUNK)))
    similarity_top_k = int(os.getenv('SIMILARITY_TOP_K', str(DEFAULT_SIMILARITY_TOP_K)))
    
    print(f"  - Number of markdown files: {num_samples}")
    print(f"  - Number of nodes to process: ALL (no limit)")
    print(f"  - Chunk size: {chunk_size} (optimized for GPT-4o-mini)")
    print(f"  - Chunk overlap: {chunk_overlap}")
    print(f"  - Max paths per chunk: {max_paths_per_chunk} (increased for better extraction)")
    print(f"  - Similarity top K: {similarity_top_k}")
    print(f"  - Model: {DEFAULT_MODEL} (128K context window)")
    
    try:
        total_steps = 10
        
        # Step 1: Convert HTML files to Markdown
        print_progress("Converting HTML files to Markdown...", 1, total_steps)
        conversion_success = convert_html_to_markdown()
        
        if not conversion_success:
            print("⚠️ HTML to Markdown conversion failed, but continuing with existing files...")
        
        # Step 2: Load data
        print_progress("Loading markdown files...", 2, total_steps)
        documents, files_df = load_data(num_samples)
        
        if not documents:
            print("❌ Failed to load documents")
            return False
        
        print(f"✅ Loaded {len(documents)} markdown documents")
        
        # Step 3: Setup LLM
        print_progress("Setting up LLM...", 3, total_steps)
        llm = setup_llm()
        
        if not llm:
            print("❌ Failed to setup LLM")
            return False
        
        print("✅ LLM setup successful")
        
        # Step 4: Setup embedding model
        print_progress("Setting up embedding model...", 4, total_steps)
        embed_model = setup_embedding_model()
        
        if not embed_model:
            print("❌ Failed to setup embedding model")
            return False
        
        print("✅ Embedding model setup successful")
        
        # Step 5: Create nodes
        print_progress("Creating text nodes...", 5, total_steps)
        nodes = create_nodes(documents, chunk_size, chunk_overlap)
        
        if not nodes:
            print("❌ Failed to create nodes")
            return False
        
        print(f"✅ Created {len(nodes)} nodes")
        
        # Step 6: Setup KG extractor
        print_progress("Setting up KG extractor...", 6, total_steps)
        kg_extractor = setup_kg_extractor(llm, max_paths_per_chunk)
        
        if not kg_extractor:
            print("❌ Failed to setup KG extractor")
            return False
        
        print("✅ KG extractor setup successful")
        
        # Step 7: Setup graph store
        print_progress("Setting up graph store...", 7, total_steps)
        graph_store = setup_graph_store(llm)
        
        if not graph_store:
            print("❌ Failed to setup graph store")
            return False
        
        print("✅ Graph store setup successful")
        
        # Step 8: Build index
        print_progress("Building index...", 8, total_steps)
        index = build_index(nodes, kg_extractor, graph_store, embed_model, num_nodes)
        
        if not index:
            print("❌ Failed to build index")
            return False
        
        print("✅ Index built successfully")
        
        # Step 9: Build communities
        print_progress("Building communities...", 9, total_steps)
        communities_success = build_communities(index)
        
        if communities_success:
            print("✅ Communities built successfully")
        else:
            print("⚠️ Community building had issues, but continuing...")
        
        # Step 10: Setup query engine
        print_progress("Setting up query engine...", 10, total_steps)
        query_engine = setup_query_engine(index, llm, similarity_top_k)
        
        if not query_engine:
            print("❌ Failed to setup query engine")
            return False
        
        print("✅ Query engine setup successful")
        
        # Save index data
        print_progress("Saving index data...")
        save_success = save_index_data(index, query_engine, files_df)
        
        if not save_success:
            print("⚠️ Failed to save index data, but index is built in Neo4j")
        
        # Final summary
        print("\n🎉 GraphRAG Index Building Completed Successfully!")
        print("=" * 50)
        
        # Get some statistics
        try:
            triplets = index.property_graph_store.get_triplets()
            print(f"📊 Statistics:")
            print(f"  - Total triplets in graph: {len(triplets)}")
            print(f"  - Documents processed: {len(documents)}")
            print(f"  - Nodes created: {len(nodes)}")
            if num_nodes is None:
                print(f"  - Nodes processed: ALL ({len(nodes)})")
            else:
                print(f"  - Nodes processed: {min(num_nodes, len(nodes))}")
            
            if hasattr(index.property_graph_store, 'get_community_summaries'):
                communities = index.property_graph_store.get_community_summaries()
                print(f"  - Communities built: {len(communities)}")
        except Exception as e:
            print(f"⚠️ Could not get statistics: {e}")
        
        print(f"\n✅ Index is ready! You can now run the Streamlit app:")
        print(f"   streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 