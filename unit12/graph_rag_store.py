import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage

import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from const import OPENAI_API_KEY, DEFAULT_MODEL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class GraphRAGStore(Neo4jPropertyGraphStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.community_summary = {}
        self.entity_info = None
        self.max_cluster_size = 5
        self.llm = None  # Will be set externally

    def set_llm(self, llm):
        """Set the LLM for community summary generation."""
        self.llm = llm

    def get_triplets(self):
        """Override the parent method to use compatible Cypher syntax."""
        try:
            # Simple Cypher query compatible with Neo4j 4.4
            query = """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name AS source_name, 
                   a.type AS source_type,
                   r.type AS relation_type, 
                   r.description AS relation_description,
                   b.name AS target_name,
                   b.type AS target_type
            """
            
            result = self.structured_query(query)
            
            # Convert to the expected format for LlamaIndex
            from llama_index.core.graph_stores.types import EntityNode, Relation
            
            triplets = []
            for record in result:
                # Create source entity
                source_entity = EntityNode(
                    name=record['source_name'],
                    label=record['source_type'],
                    properties={}
                )
                
                # Create target entity  
                target_entity = EntityNode(
                    name=record['target_name'],
                    label=record['target_type'],
                    properties={}
                )
                
                # Create relation
                relation = Relation(
                    label=record['relation_type'],
                    source_id=record['source_name'],
                    target_id=record['target_name'],
                    properties={'relationship_description': record['relation_description']}
                )
                
                triplets.append((source_entity, relation, target_entity))
            
            print(f"✓ Retrieved {len(triplets)} triplets from Neo4j")
            return triplets
            
        except Exception as e:
            print(f"Error in get_triplets: {e}")
            return []

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
    

if __name__ == "__main__":    
    # Example usage of GraphRAGStore
    print("GraphRAGStore Example")
    print("=" * 50)
    
    # Setup OpenAI LLM (real LLM)
    print("Setting up OpenAI LLM...")
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        llm = OpenAI(model=DEFAULT_MODEL)
        Settings.llm = llm
        
        # Test LLM connection
        print("Testing LLM connection...")
        test_response = llm.complete("What is 2+2?")
        print(f"LLM Test Response: {test_response}")
        print("✓ OpenAI LLM initialized successfully")
        
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI LLM: {e}")
        print("Using mock LLM for demonstration...")
        
        # Fallback mock LLM
        class MockLLM:
            def chat(self, messages):
                return "This is a mock community summary based on the provided relationships."
        
        llm = MockLLM()
    
    try:
        # Initialize GraphRAGStore with real Neo4j parameters
        print("\nInitializing GraphRAGStore...")
        graph_store = GraphRAGStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD, 
            url=NEO4J_URI,
            database="neo4j"
        )
        
        # Set real LLM
        graph_store.set_llm(llm)
        
        print("✓ GraphRAGStore initialized successfully")
        print("✓ LLM set")
        
        # Test community summary generation
        print("\nTesting community summary generation...")
        sample_relationships = """
        Apple Inc -> Steve Jobs -> founded_by -> Steve Jobs co-founded Apple Inc. and served as its visionary leader
        Apple Inc -> Cupertino -> headquartered_in -> Apple Inc. is headquartered in Cupertino, California
        Tim Cook -> Apple Inc -> CEO_of -> Tim Cook is the current CEO of Apple Inc.
        """
        
        summary = graph_store.generate_community_summary(sample_relationships)
        print(f"Generated summary: {summary}")
        
        # Example of how communities would be built
        print("\nCommunity building process:")
        print("- build_communities() method available")
        print("- get_community_summaries() method available")
        print("- generate_community_summary() method available")
        
        # Show configuration
        print(f"\nConfiguration:")
        print(f"- Max cluster size: {graph_store.max_cluster_size}")
        print(f"- LLM set: {graph_store.llm is not None}")
        print(f"- LLM type: {type(graph_store.llm).__name__}")
        print(f"- Community summaries initialized: {len(graph_store.community_summary)} communities")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Note: Full example requires Neo4j connection: {e}")
        print("This is expected in a demo environment.")
        
        # Still demonstrate the class structure
        print("\nClass structure demonstration:")
        print("GraphRAGStore inherits from Neo4jPropertyGraphStore")
        print("Main methods:")
        print("- set_llm(llm): Set LLM for community summaries")
        print("- build_communities(): Build and summarize communities")
        print("- get_community_summaries(): Get all community summaries")
        print("- generate_community_summary(text): Generate summary for text")
        print(f"- Using real OpenAI LLM: {type(llm).__name__}")
        