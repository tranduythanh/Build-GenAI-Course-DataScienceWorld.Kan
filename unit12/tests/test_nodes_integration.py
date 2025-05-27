#!/usr/bin/env python3
"""
Test script to verify nodes (chunks) integration
"""

import os
import sys

# Add parent directory to path so we can import from parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_index_data, load_nodes

def test_nodes_integration():
    """Test if nodes are properly saved and loaded."""
    print("🧪 Testing Nodes Integration")
    print("=" * 50)
    
    # Test 1: Check if index data exists
    print("\n📋 Test 1: Checking index data...")
    metadata, files_df, nodes = load_index_data()
    
    if metadata is None:
        print("❌ No index data found. Please run 'python build_index.py' first.")
        return False
    
    print(f"✅ Index metadata loaded:")
    print(f"   - Built: {metadata.get('timestamp', 'Unknown')}")
    print(f"   - Files processed: {metadata.get('files_processed', 'Unknown')}")
    print(f"   - Nodes count: {metadata.get('nodes_count', 'Unknown')}")
    
    # Test 2: Check if nodes are loaded
    print("\n📄 Test 2: Checking nodes...")
    if nodes is None:
        print("❌ No nodes found in index data.")
        
        # Try loading nodes separately
        print("🔍 Trying to load nodes separately...")
        nodes = load_nodes()
        
        if nodes is None:
            print("❌ No nodes.pkl file found. Nodes were not saved during build.")
            return False
    
    print(f"✅ Nodes loaded successfully:")
    print(f"   - Total nodes: {len(nodes)}")
    
    # Test 3: Examine node structure
    print("\n🔍 Test 3: Examining node structure...")
    if len(nodes) > 0:
        sample_node = nodes[0]
        print(f"✅ Sample node structure:")
        print(f"   - Type: {type(sample_node)}")
        print(f"   - Has text: {hasattr(sample_node, 'text')}")
        print(f"   - Has metadata: {hasattr(sample_node, 'metadata')}")
        
        if hasattr(sample_node, 'text'):
            print(f"   - Text length: {len(sample_node.text)}")
            print(f"   - Text preview: {sample_node.text[:100]}...")
        
        if hasattr(sample_node, 'metadata'):
            print(f"   - Metadata keys: {list(sample_node.metadata.keys()) if sample_node.metadata else 'None'}")
    
    # Test 4: Check file sizes
    print("\n💾 Test 4: Checking file sizes...")
    index_dir = "./index_data"
    
    files_to_check = [
        "index_metadata.pkl",
        "files_df.pkl", 
        "nodes.pkl"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(index_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename}: {size:,} bytes")
        else:
            print(f"❌ {filename}: Not found")
    
    print("\n🎉 Nodes integration test completed!")
    return True

def test_query_engine_with_nodes():
    """Test if query engine can use nodes."""
    print("\n🔧 Testing Query Engine with Nodes")
    print("=" * 50)
    
    try:
        from utils import setup_llm, setup_graph_store, setup_query_engine
        from llama_index.core import PropertyGraphIndex
        
        # Load nodes
        metadata, files_df, nodes = load_index_data()
        
        if nodes is None:
            print("❌ No nodes available for testing")
            return False
        
        print(f"✅ Loaded {len(nodes)} nodes for testing")
        
        # Setup components
        print("🔧 Setting up LLM...")
        llm = setup_llm()
        if not llm:
            print("❌ Failed to setup LLM")
            return False
        
        print("🔧 Setting up graph store...")
        graph_store = setup_graph_store(llm)
        if not graph_store:
            print("❌ Failed to setup graph store")
            return False
        
        print("🔧 Creating index...")
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_kg_nodes=False
        )
        
        print("🔧 Setting up query engine with nodes...")
        query_engine = setup_query_engine(index, llm, 10, nodes)
        
        if not query_engine:
            print("❌ Failed to setup query engine")
            return False
        
        print("✅ Query engine setup successful with nodes!")
        print(f"   - Query engine has nodes: {hasattr(query_engine, 'nodes')}")
        print(f"   - Nodes count in query engine: {len(query_engine.nodes) if hasattr(query_engine, 'nodes') and query_engine.nodes else 0}")
        
        # Test chunk retrieval
        print("\n🔍 Testing chunk retrieval...")
        test_query = "What are the main components of LLM agents?"
        test_entities = ["LLM", "agent", "component"]
        
        relevant_chunks = query_engine.get_relevant_chunks(test_query, test_entities)
        print(f"✅ Found {len(relevant_chunks)} relevant chunks for test query")
        
        if relevant_chunks:
            print(f"   - Top chunk score: {relevant_chunks[0]['score']}")
            print(f"   - Top chunk preview: {relevant_chunks[0]['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing query engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 GraphRAG Nodes Integration Test")
    print("=" * 60)
    
    # Run tests
    test1_success = test_nodes_integration()
    
    if test1_success:
        test2_success = test_query_engine_with_nodes()
        
        if test1_success and test2_success:
            print("\n🎉 All tests passed! Nodes integration is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
    else:
        print("\n❌ Basic nodes test failed. Please rebuild the index with 'python build_index.py'")
        sys.exit(1) 