import re
import json
import os
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from const import OPENAI_API_KEY, DEFAULT_MODEL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, DEFAULT_COMMUNITY_FOLDER, DEFAULT_MAX_CLUSTER_SIZE

class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    GraphRAG Store extending Neo4jPropertyGraphStore with community detection and caching.
    
    Default Configuration:
    - Community summaries saved to: community/summary.json
    - Max cluster size: 5 (from DEFAULT_MAX_CLUSTER_SIZE)
    - Auto-saves community summaries after building
    - Auto-loads from cache if available
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.community_summary = {}
        self.entity_info = None
        self.max_cluster_size = DEFAULT_MAX_CLUSTER_SIZE  # Use constant from config
        self.llm = None  # Will be set externally
        self.data_folder = DEFAULT_COMMUNITY_FOLDER  # Use constant for consistency

    def set_llm(self, llm):
        """Set the LLM for community summary generation."""
        self.llm = llm

    def set_community_folder(self, folder_path: str):
        """Set the folder path for saving/loading community data."""
        self.data_folder = folder_path
        print(f"ℹ️ Community data folder set to: {folder_path}")

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
        print("🏗️ Building communities from Neo4j graph...")
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)
        print(f"✅ Built {len(self.community_summary)} communities with summaries")
        
        # Automatically save to file after building
        self.save_community_summaries_to_file()

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
        """Returns the community summaries, loading from file or building them if not already done."""
        print(f"🔍 get_community_summaries called. Current summaries count: {len(self.community_summary)}")
        print(f"🔍 entity_info status: {type(self.entity_info)} with {len(self.entity_info) if self.entity_info else 0} entities")
        
        # First try to load from file if we don't have summaries yet
        if not self.community_summary:
            # Try to load from file first
            if self.load_community_summaries_from_file():
                print("✅ Using community summaries from file")
                return self.community_summary
            else:
                print("ℹ️ Building communities from Neo4j graph...")
                # Build from Neo4j if file doesn't exist
                self.build_communities()
                # Save to file for future use
                self.save_community_summaries_to_file()
        
        return self.community_summary
    
    def initialize_from_cache(self):
        """Initialize community data from cache during startup."""
        print("🚀 Initializing community data from cache...")
        if self.load_community_summaries_from_file():
            print(f"✅ Initialized with {len(self.community_summary)} communities and {len(self.entity_info)} entities from cache")
            return True
        else:
            print("ℹ️ No cache found, will build communities when needed")
            return False
    
    def load_community_summaries_from_file(self):
        """Load community summaries from summary.json if it exists."""
        community_file = os.path.join(self.data_folder, "summary.json")
        
        print(f"🔍 Loading community summaries from: {community_file}")
        print(f"🔍 File exists: {os.path.exists(community_file)}")
        
        if os.path.exists(community_file):
            try:
                with open(community_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.community_summary = data.get('community_summaries', {})
                    self.entity_info = data.get('entity_info', {})
                
                print(f"✅ Loaded {len(self.community_summary)} community summaries from {community_file}")
                print(f"✅ Loaded entity_info for {len(self.entity_info)} entities")
                print(f"🔍 Sample entity_info keys: {list(self.entity_info.keys())[:5]}")
                print(f"🔍 entity_info type: {type(self.entity_info)}")
                return True
            except Exception as e:
                print(f"❌ Error loading community summaries from file: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"ℹ️ Community summary file not found: {community_file}")
            return False
    
    def save_community_summaries_to_file(self):
        """Save community summaries to summary.json."""
        # Ensure data folder exists
        os.makedirs(self.data_folder, exist_ok=True)
        
        community_file = os.path.join(self.data_folder, "summary.json")
        
        try:
            data = {
                'community_summaries': self.community_summary,
                'entity_info': self.entity_info,
                'metadata': {
                    'total_communities': len(self.community_summary),
                    'total_entities': len(self.entity_info) if self.entity_info else 0,
                    'max_cluster_size': self.max_cluster_size
                }
            }
            
            with open(community_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Saved {len(self.community_summary)} community summaries to {community_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving community summaries to file: {e}")
            return False

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
        