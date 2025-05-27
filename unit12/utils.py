#!/usr/bin/env python3
"""
Shared utility functions for GraphRAG application.
These functions are used by both app.py and build_index.py.
"""

import os
import pickle
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
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



def setup_query_engine(index, llm, similarity_top_k, nodes=None):
    """Setup the GraphRAG query engine."""
    try:
        query_engine = GraphRAGQueryEngine(
            graph_store=index.property_graph_store,
            llm=llm,
            index=index,
            similarity_top_k=similarity_top_k,
            nodes=nodes
        )
        return query_engine
    except Exception as e:
        print(f"❌ Error setting up query engine: {e}")
        return None


def load_index_data(data_dir="./index_data"):
    """Load pre-built index data from disk."""
    try:
        metadata_path = os.path.join(data_dir, 'index_metadata.pkl')
        files_df_path = os.path.join(data_dir, 'files_df.pkl')
        nodes_path = os.path.join(data_dir, 'nodes.pkl')
        
        if not os.path.exists(metadata_path):
            return None, None, None
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load files dataframe if exists
        files_df = None
        if os.path.exists(files_df_path):
            files_df = pd.read_pickle(files_df_path)
        
        # Load nodes if exists
        nodes = None
        if os.path.exists(nodes_path):
            with open(nodes_path, 'rb') as f:
                nodes = pickle.load(f)
            print(f"✅ Loaded {len(nodes)} nodes from disk")
        
        return metadata, files_df, nodes
        
    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Error loading index data: {e}")
        else:
            print(f"Error loading index data: {e}")
        return None, None, None


def load_nodes(data_dir="./index_data"):
    """Load nodes from disk."""
    try:
        nodes_path = os.path.join(data_dir, 'nodes.pkl')
        
        if not os.path.exists(nodes_path):
            return None
        
        with open(nodes_path, 'rb') as f:
            nodes = pickle.load(f)
        
        print(f"✅ Loaded {len(nodes)} nodes from {nodes_path}")
        return nodes
        
    except Exception as e:
        print(f"❌ Error loading nodes: {e}")
        return None


def check_neo4j_connection(graph_store):
    """Check if Neo4j has data and is accessible."""
    try:
        triplets = graph_store.get_triplets()
        return len(triplets) > 0, len(triplets)
    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Neo4j connection error: {e}")
        else:
            print(f"Neo4j connection error: {e}")
        return False, 0


def create_triplets_graph(triplets, max_nodes=50):
    """Create an interactive graph visualization of triplets using Plotly."""
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges from triplets (limit for performance)
        triplets_to_show = triplets[:max_nodes] if len(triplets) > max_nodes else triplets
        
        for entity1, relation, entity2 in triplets_to_show:
            source = entity1.name if hasattr(entity1, 'name') else str(entity1)
            target = entity2.name if hasattr(entity2, 'name') else str(entity2)
            rel_label = relation.label if hasattr(relation, 'label') else str(relation)
            
            G.add_edge(source, target, relation=rel_label)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(G[edge[0]][edge[1]]['relation'])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Count connections
            adjacencies = list(G.neighbors(node))
            node_info.append(f'{node}<br>Connections: {len(adjacencies)}')
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=[],
                size=15,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Node Connections"
                ),
                line=dict(width=2)
            )
        )
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Knowledge Graph Visualization ({len(triplets_to_show)} triplets)',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes to see connections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        return fig
        
    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Error creating triplets graph: {e}")
        else:
            print(f"Error creating triplets graph: {e}")
        return None


def create_communities_graph(graph_store):
    """Create a visualization of communities in the knowledge graph."""
    try:
        # Get communities
        if not hasattr(graph_store, 'get_community_summaries'):
            return None
            
        communities = graph_store.get_community_summaries()
        if not communities:
            # Try to build communities first
            graph_store.build_communities()
            communities = graph_store.get_community_summaries()
            
        if not communities:
            return None
        
        # Get entity info for community membership
        entity_info = getattr(graph_store, 'entity_info', {})
        if not entity_info:
            return None
        
        # Create NetworkX graph for communities
        G = nx.Graph()
        
        # Color map for communities
        colors = px.colors.qualitative.Set3
        community_colors = {}
        
        # Add nodes with community colors
        for entity, community_list in entity_info.items():
            if community_list:
                primary_community = community_list[0]  # Use first community as primary
                if primary_community not in community_colors:
                    community_colors[primary_community] = colors[len(community_colors) % len(colors)]
                
                G.add_node(entity, community=primary_community, color=community_colors[primary_community])
        
        # Add edges between entities in the same communities
        triplets = graph_store.get_triplets()
        for entity1, relation, entity2 in triplets:
            source = entity1.name if hasattr(entity1, 'name') else str(entity1)
            target = entity2.name if hasattr(entity2, 'name') else str(entity2)
            
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create traces for each community
        traces = []
        
        for community_id, color in community_colors.items():
            # Get nodes in this community
            community_nodes = [node for node, data in G.nodes(data=True) 
                             if data.get('community') == community_id]
            
            if not community_nodes:
                continue
                
            node_x = [pos[node][0] for node in community_nodes]
            node_y = [pos[node][1] for node in community_nodes]
            
            trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=community_nodes,
                textposition="middle center",
                name=f'Community {community_id}',
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>Community: ' + str(community_id) + '<extra></extra>'
            )
            traces.append(trace)
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        traces.insert(0, edge_trace)  # Add edges first so they appear behind nodes
        
        # Create figure
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title=dict(
                               text=f'Community Structure ({len(community_colors)} communities)',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600,
                           legend=dict(
                               yanchor="top",
                               y=0.99,
                               xanchor="left",
                               x=1.01
                           )
                       ))
        
        return fig
        
    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Error creating communities graph: {e}")
        else:
            print(f"Error creating communities graph: {e}")
        return None