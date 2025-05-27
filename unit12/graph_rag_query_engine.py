import re
import time
import os

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.llms.openai import OpenAI
from graph_rag_store import GraphRAGStore
from const import (
    OPENAI_API_KEY, 
    DEFAULT_MODEL, 
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    DEFAULT_SIMILARITY_TOP_K
)

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        """Extract entities relevant to the query by searching the graph store directly."""
        entities = set()
        
        try:
            # First try the original retriever approach
            nodes_retrieved = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            ).retrieve(query_str)

            # Try to extract entities from node text using the original pattern
            pattern = (
                r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
            )

            for node in nodes_retrieved:
                matches = re.findall(
                    pattern, node.text, re.MULTILINE | re.IGNORECASE
                )

                for match in matches:
                    subject = match[0]
                    obj = match[2]
                    entities.add(subject)
                    entities.add(obj)
            
            # If no entities found from retriever, search Neo4j directly
            if not entities:
                entities = self._search_entities_in_graph(query_str, similarity_top_k)
                
        except Exception as e:
            print(f"Error in retriever, falling back to direct graph search: {e}")
            entities = self._search_entities_in_graph(query_str, similarity_top_k)

        print(f"✓ Found {len(entities)} entities for query: '{query_str}'")
        return list(entities)
    
    def _search_entities_in_graph(self, query_str, top_k):
        """Search for entities in the graph store based on query terms."""
        entities = set()
        
        try:
            # Extract meaningful words from the query (remove common stop words)
            stop_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 
                         'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'about', 'tell', 'me', 'their', 'them', 'this', 'that'}
            
            query_words = [word.strip().lower() for word in query_str.split() 
                          if word.strip().lower() not in stop_words and len(word.strip()) > 2]
            
            if not query_words:
                return entities
            
            # Search for entities whose names or descriptions contain query terms
            for word in query_words:
                # Use CONTAINS for partial matching (case-insensitive)
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS $word 
                   OR toLower(e.description) CONTAINS $word
                   OR toLower(e.type) CONTAINS $word
                RETURN DISTINCT e.name AS entity_name
                LIMIT $limit
                """
                
                result = self.graph_store.structured_query(
                    cypher_query, 
                    param_map={"word": word, "limit": top_k}
                )
                
                for record in result:
                    entities.add(record['entity_name'])
            
            # Also search for entities connected to entities that match query terms
            if entities:
                entity_list = list(entities)[:5]  # Limit to avoid too large queries
                cypher_query = """
                MATCH (e1:Entity)-[r:RELATION]-(e2:Entity)
                WHERE e1.name IN $entity_names OR e2.name IN $entity_names
                RETURN DISTINCT e1.name AS entity1, e2.name AS entity2
                LIMIT $limit
                """
                
                result = self.graph_store.structured_query(
                    cypher_query,
                    param_map={"entity_names": entity_list, "limit": top_k * 2}
                )
                
                for record in result:
                    entities.add(record['entity1'])
                    entities.add(record['entity2'])
                    
        except Exception as e:
            print(f"Error searching entities in graph: {e}")
            
        return entities

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        time.sleep(1)  # Reduced sleep time for demo
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

if __name__ == "__main__":
    # Setup OpenAI LLM
    print("GraphRAGQueryEngine Example")
    print("=" * 50)
    
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
        print(f"Error initializing OpenAI LLM: {e}")
        print("Cannot proceed without a valid LLM.")
        exit(1)
    
    # Manual Neo4j setup
    import neo4j
    try:
        print("\nConnecting to Neo4j and setting up sample data...")
        
        # Direct Neo4j connection
        driver = neo4j.GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        # Clear existing data and create sample graph
        with driver.session() as session:
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            print("Creating sample entities and relationships...")
            
            # Create entities
            entities_data = [
                {"name": "Apple Inc", "type": "Organization", "description": "Technology company that designs consumer electronics"},
                {"name": "Steve Jobs", "type": "Person", "description": "Co-founder and former CEO of Apple Inc"},
                {"name": "Tim Cook", "type": "Person", "description": "Current CEO of Apple Inc"},
                {"name": "Cupertino", "type": "Location", "description": "City in California where Apple is headquartered"},
                {"name": "Microsoft Corporation", "type": "Organization", "description": "Software company developing Windows and other products"},
                {"name": "Bill Gates", "type": "Person", "description": "Co-founder of Microsoft Corporation"},
                {"name": "Satya Nadella", "type": "Person", "description": "Current CEO of Microsoft Corporation"},
                {"name": "Tesla", "type": "Organization", "description": "Electric vehicle manufacturer"},
                {"name": "Elon Musk", "type": "Person", "description": "CEO of Tesla and SpaceX"},
                {"name": "Austin", "type": "Location", "description": "City in Texas where Tesla is headquartered"}
            ]
            
            for entity in entities_data:
                session.run(
                    "CREATE (e:Entity {name: $name, type: $type, description: $description})",
                    entity
                )
            
            # Create relationships  
            relationships = [
                ("Steve Jobs", "Apple Inc", "FOUNDED", "Steve Jobs co-founded Apple Inc"),
                ("Tim Cook", "Apple Inc", "CEO_OF", "Tim Cook is the current CEO of Apple Inc"),
                ("Apple Inc", "Cupertino", "HEADQUARTERED_IN", "Apple Inc is headquartered in Cupertino"),
                ("Bill Gates", "Microsoft Corporation", "FOUNDED", "Bill Gates co-founded Microsoft Corporation"),
                ("Satya Nadella", "Microsoft Corporation", "CEO_OF", "Satya Nadella is the current CEO of Microsoft"),
                ("Elon Musk", "Tesla", "CEO_OF", "Elon Musk is the CEO of Tesla"),
                ("Tesla", "Austin", "HEADQUARTERED_IN", "Tesla is headquartered in Austin")
            ]
            
            for source, target, rel_type, description in relationships:
                session.run("""
                    MATCH (s:Entity {name: $source})
                    MATCH (t:Entity {name: $target})
                    CREATE (s)-[r:RELATION {type: $rel_type, description: $description}]->(t)
                """, {"source": source, "target": target, "rel_type": rel_type, "description": description})
        
        print("✓ Sample data created in Neo4j successfully")
        
        # Initialize GraphRAGStore
        print("\nInitializing GraphRAGStore...")
        graph_store = GraphRAGStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j"
        )
        graph_store.set_llm(llm)
        print("✓ GraphRAGStore initialized successfully")
        
        # Build communities manually since we have real data now
        print("Building communities from existing data...")
        graph_store.build_communities()
        community_summaries = graph_store.get_community_summaries()
        print(f"✓ Built {len(community_summaries)} communities")
        
        # Create PropertyGraphIndex
        print("Creating PropertyGraphIndex...")
        property_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_kg_nodes=False,  # Disable embedding to avoid vector similarity issues
        )
        print("✓ PropertyGraphIndex created successfully")
        
        # Initialize GraphRAGQueryEngine
        print("\nInitializing GraphRAGQueryEngine...")
        query_engine = GraphRAGQueryEngine(
            graph_store=graph_store,
            index=property_index,
            llm=llm,
            similarity_top_k=DEFAULT_SIMILARITY_TOP_K
        )
        print("✓ GraphRAGQueryEngine initialized successfully")
        
        # Test queries
        test_queries = [
            "Who founded Apple Inc?",
            "What companies are mentioned and their CEOs?",
            "Tell me about technology companies and their headquarters.",
            "What is the relationship between Steve Jobs and Apple?"
        ]
        
        print(f"\nTesting {len(test_queries)} sample queries...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 40)
            
            try:
                # Execute query
                print("Processing query...")
                response = query_engine.custom_query(query)
                print(f"Response: {response}")
                
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
            
            print("\n" + "="*50)
        
        print(f"\nGraphRAGQueryEngine example completed successfully!")
        
        # Summary statistics  
        print(f"\nSummary:")
        print(f"Total communities: {len(community_summaries)}")
        print(f"Similarity top-k: {query_engine.similarity_top_k}")
        print(f"Sample queries tested: {len(test_queries)}")
        
        # Show some community summaries
        print(f"\nCommunity summaries:")
        for i, (community_id, summary) in enumerate(community_summaries.items()):
            if i < 3:  # Show first 3
                print(f"Community {community_id}: {summary[:100]}...")
        
        driver.close()
        
    except Exception as e:
        print(f"Error in main example: {e}")
        import traceback
        traceback.print_exc()
        
        # Show what could be done without full setup
        print("\nNote: This example requires:")
        print("- Valid OpenAI API key")
        print("- Running Neo4j database")
        print("- All dependencies installed")
        print("\nWithout these, you can still use the GraphRAGQueryEngine class")
        print("by providing your own graph_store, index, and llm instances.")