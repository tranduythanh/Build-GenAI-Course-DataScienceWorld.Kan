#!/usr/bin/env python3
"""
Build-specific utility functions for GraphRAG index building.
These functions are only used by build_index.py.
"""

import os
import glob
import pandas as pd
from typing import Any

# Local imports
from const import (
    CSV_KG_EXTRACT_TMPL,
    EMBEDDING_MODEL,
)
from graph_rag_extractor import GraphRAGExtractor, parse_csv_triplets_fn
from html_to_md_converter import HTMLToMarkdownConverter

# LlamaIndex imports
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY


def convert_html_to_markdown():
    """Convert all HTML files in data folder to Markdown."""
    try:
        # Get all HTML files from data folder
        html_files = glob.glob("data/*.html")
        
        if not html_files:
            print("ℹ️ No HTML files found in data folder to convert")
            return True
        
        converter = HTMLToMarkdownConverter()
        converted_files = []
        
        for html_file in html_files:
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(html_file))[0]
                md_file = f"data/{base_name}_converted.md"
                
                # Skip if markdown file already exists and is newer
                if os.path.exists(md_file):
                    html_mtime = os.path.getmtime(html_file)
                    md_mtime = os.path.getmtime(md_file)
                    if md_mtime > html_mtime:
                        print(f"ℹ️ Skipping {html_file} - Markdown file is up to date")
                        continue
                
                # Convert HTML to Markdown
                print(f"ℹ️ Converting {html_file} to {md_file}")
                converter.convert_file(html_file, md_file)
                converted_files.append(md_file)
                
            except Exception as file_error:
                print(f"⚠️ Error converting {html_file}: {file_error}")
                continue
        
        if converted_files:
            print(f"✅ Converted {len(converted_files)} HTML files to Markdown")
        else:
            print("ℹ️ All HTML files are already converted and up to date")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during HTML to Markdown conversion: {e}")
        return False


def load_data(num_samples: int = 50):
    """Load and prepare markdown documents from data folder."""
    try:
        # Get all markdown files from data folder
        markdown_files = glob.glob("data/*.md")
        
        if not markdown_files:
            print("❌ No markdown files found in data folder")
            return [], pd.DataFrame()
        
        documents = []
        file_info = []
        
        # Limit the number of files processed based on num_samples
        files_to_process = markdown_files[:num_samples] if num_samples < len(markdown_files) else markdown_files
        
        for file_path in files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract filename for metadata
                filename = os.path.basename(file_path)
                
                # Create document with filename as title
                documents.append(
                    Document(
                        text=content,
                        metadata={"source": filename, "file_path": file_path}
                    )
                )
                
                # Store file info for display
                file_info.append({
                    "filename": filename,
                    "file_path": file_path,
                    "content_length": len(content),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })
                
            except Exception as file_error:
                print(f"⚠️ Error reading file {file_path}: {file_error}")
                continue
        
        # Create DataFrame for display purposes
        files_df = pd.DataFrame(file_info)
        
        return documents, files_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return [], pd.DataFrame()


def setup_embedding_model():
    """Setup the embedding model."""
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        return embed_model
    except Exception as e:
        print(f"❌ Error setting up embedding model: {e}")
        return None


def create_nodes(documents, chunk_size, chunk_overlap):
    """Create text nodes/chunks from documents."""
    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes
    except Exception as e:
        print(f"❌ Error creating nodes: {e}")
        return []


def setup_kg_extractor(llm, max_paths_per_chunk):
    """Setup the Knowledge Graph extractor."""
    try:
        # Try using our custom GraphRAGExtractor first
        kg_extractor = GraphRAGExtractor(
            llm=llm,
            extract_prompt=CSV_KG_EXTRACT_TMPL,
            max_paths_per_chunk=max_paths_per_chunk,
            parse_fn=parse_csv_triplets_fn,
        )
        return kg_extractor
    except Exception as e:
        print(f"⚠️ Custom KG extractor failed: {e}")
        
        # Fallback to LlamaIndex's built-in extractor
        try:
            from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
            
            print("ℹ️ Using SimpleLLMPathExtractor as fallback...")
            kg_extractor = SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=max_paths_per_chunk,
                num_workers=1  # Reduce workers to avoid issues
            )
            return kg_extractor
        except Exception as fallback_error:
            print(f"❌ Fallback KG extractor also failed: {fallback_error}")
            return None


def build_index(nodes, kg_extractor, graph_store, embed_model, num_nodes):
    """Build the Property Graph Index using a compatible approach."""
    try:
        # Determine how many nodes to process
        if num_nodes is None:
            nodes_to_process = len(nodes)
            processed_nodes = nodes
            print(f"ℹ️ Building PropertyGraphIndex with ALL {nodes_to_process} nodes...")
        else:
            nodes_to_process = min(num_nodes, len(nodes))
            processed_nodes = nodes[:num_nodes]
            print(f"ℹ️ Building PropertyGraphIndex with {nodes_to_process} nodes...")
        
        # Clear existing data first
        print("ℹ️ Clearing existing Neo4j data...")
        graph_store.structured_query("MATCH (n) DETACH DELETE n")
        
        # Use from_existing approach to avoid Neo4j syntax issues
        print("ℹ️ Creating PropertyGraphIndex from existing store...")
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=embed_model,
            embed_kg_nodes=False,  # Disable embedding to avoid vector issues
            show_progress=False
        )
        
        # Manually process nodes and extract knowledge
        print(f"ℹ️ Processing {len(processed_nodes)} nodes...")
        
        # Extract knowledge from each node using the kg_extractor
        for i, node in enumerate(processed_nodes):
            try:
                print(f"\n{'='*80}")
                print(f"🔍 Processing node {i+1}/{len(processed_nodes)}")
                print(f"{'='*80}")
                
                # Print node text with brown color
                node_text = node.text
                print(f"\033[38;5;94m📄 Node Text:\033[0m")
                print(f"\033[38;5;137m{node_text}\033[0m")
                print()
                
                # Extract triplets from the node text
                entities_list = []
                relations_list = []
                triplets = []
                
                if hasattr(kg_extractor, '__call__') and hasattr(kg_extractor, 'parse_fn'):
                    # Custom GraphRAGExtractor - expects list of nodes
                    extracted_nodes = kg_extractor([node], show_progress=False)
                    
                    for extracted_node in extracted_nodes:
                        # Get entities and relations from metadata
                        entities = extracted_node.metadata.get(KG_NODES_KEY, [])
                        relations = extracted_node.metadata.get(KG_RELATIONS_KEY, [])
                        
                        entities_list.extend(entities)
                        relations_list.extend(relations)
                        
                        # Convert to triplet format for Neo4j insertion
                        for relation in relations:
                            triplets.append((relation.source_id, relation.label, relation.target_id))
                else:
                    # LlamaIndex SimpleLLMPathExtractor
                    extracted_data = kg_extractor.extract([node])
                    for extracted_node in extracted_data:
                        if hasattr(extracted_node, 'metadata') and 'kg_triplets' in extracted_node.metadata:
                            triplets.extend(extracted_node.metadata['kg_triplets'])
                
                # Print extraction summary
                print(f"📊 Extraction Summary:")
                print(f"   • Entities: {len(entities_list)}")
                print(f"   • Relations: {len(relations_list)}")
                print(f"   • Triplets: {len(triplets)}")
                print()
                
                # Print entities with brown color
                if entities_list:
                    print(f"\033[38;5;94m🏷️  Entities:\033[0m")
                    for j, entity in enumerate(entities_list):
                        entity_name = entity.name if hasattr(entity, 'name') else str(entity)
                        entity_type = entity.label if hasattr(entity, 'label') else 'Entity'
                        print(f"\033[38;5;137m   {j+1}. {entity_name} ({entity_type})\033[0m")
                    print()
                
                # Print relations with brown color
                if relations_list:
                    print(f"\033[38;5;94m🔗 Relations:\033[0m")
                    for j, relation in enumerate(relations_list):
                        source = relation.source_id if hasattr(relation, 'source_id') else 'Unknown'
                        label = relation.label if hasattr(relation, 'label') else 'Unknown'
                        target = relation.target_id if hasattr(relation, 'target_id') else 'Unknown'
                        print(f"\033[38;5;137m   {j+1}. {source} --[{label}]--> {target}\033[0m")
                    print()
                
                # Print all triplets with brown color
                if triplets:
                    print(f"\033[38;5;94m🔺 Triplets for Neo4j:\033[0m")
                    for j, triplet in enumerate(triplets):
                        if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                            subject, predicate, obj = triplet[0], triplet[1], triplet[2]
                        elif hasattr(triplet, 'subject') and hasattr(triplet, 'predicate') and hasattr(triplet, 'object'):
                            subject, predicate, obj = triplet.subject, triplet.predicate, triplet.object
                        else:
                            subject, predicate, obj = str(triplet), "UNKNOWN", "UNKNOWN"
                        print(f"\033[38;5;137m   {j+1}. ({subject}) --[{predicate}]--> ({obj})\033[0m")
                    print()
                
                # Manually add entities and relationships to Neo4j
                print(f"💾 Inserting {len(triplets)} triplets into Neo4j...")
                
                inserted_count = 0
                for j, triplet in enumerate(triplets):
                    try:
                        if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                            subject, predicate, obj = triplet[0], triplet[1], triplet[2]
                        elif hasattr(triplet, 'subject') and hasattr(triplet, 'predicate') and hasattr(triplet, 'object'):
                            subject, predicate, obj = triplet.subject, triplet.predicate, triplet.object
                        else:
                            print(f"\033[38;5;208m⚠️ Skipping invalid triplet format: {triplet}\033[0m")
                            continue
                        
                        # Create entities and relationship in Neo4j using simple Cypher
                        query = """
                        MERGE (s:Entity {name: $subject, type: 'Entity'})
                        MERGE (o:Entity {name: $object, type: 'Entity'})
                        MERGE (s)-[r:RELATION {type: $predicate, description: $description}]->(o)
                        """
                        
                        graph_store.structured_query(
                            query,
                            param_map={
                                "subject": str(subject),
                                "object": str(obj),
                                "predicate": str(predicate),
                                "description": f"{subject} {predicate} {obj}"
                            }
                        )
                        inserted_count += 1
                        print(f"\033[38;5;137m   ✓ Inserted: ({subject}) --[{predicate}]--> ({obj})\033[0m")
                        
                    except Exception as triplet_error:
                        print(f"\033[38;5;208m⚠️ Error processing triplet: {triplet_error}\033[0m")
                        continue
                
                print(f"\033[38;5;34m✅ Successfully inserted {inserted_count}/{len(triplets)} triplets from node {i+1}\033[0m")
                
            except Exception as node_error:
                print(f"⚠️ Error processing node {i+1}: {node_error}")
                continue
        
        print("✅ Successfully built knowledge graph in Neo4j")
        return index
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        # Try fallback approach
        try:
            print("ℹ️ Trying fallback approach...")
            index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                embed_model=embed_model,
                embed_kg_nodes=False
            )
            return index
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return None


def build_communities(index):
    """Build communities and generate summaries."""
    try:
        print("ℹ️ Building communities...")
        # Check if we have a GraphRAGStore with community building capability
        if hasattr(index.property_graph_store, 'build_communities'):
            index.property_graph_store.build_communities()
            print("✅ Communities built and summarized!")
        else:
            print("ℹ️ Using basic community structure (no advanced community detection)")
            # For basic setups, we can still proceed without communities
        return True
    except Exception as e:
        print(f"⚠️ Community building failed: {e}")
        print("ℹ️ Continuing without community detection...")
        return True  # Don't fail the entire process 