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
    nodes: list = None  # Thêm nodes gốc

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        print(f"\n🔍 Starting query processing for: '{query_str}'")
        
        # Step 1: Extract entities
        print("📋 Step 1: Extracting relevant entities...")
        entities = self.get_entities(query_str, self.similarity_top_k)
        print(f"✓ Found {len(entities)} entities: {entities[:5]}{'...' if len(entities) > 5 else ''}")

        # Step 2: Get communities
        print("🏘️ Step 2: Retrieving entity communities...")
        print(f"🔍 Debug: Entity info available: {self.graph_store.entity_info is not None}")
        if hasattr(self.graph_store, 'entity_info') and self.graph_store.entity_info:
            print(f"🔍 Debug: Entity info size: {len(self.graph_store.entity_info)}")
            print(f"🔍 Debug: Sample entities in info: {list(self.graph_store.entity_info.keys())[:5]}")
        else:
            print("⚠️ Warning: No entity_info found, attempting to build communities...")
            try:
                self.graph_store.build_communities()
                print(f"✓ Communities built, entity_info size: {len(self.graph_store.entity_info) if self.graph_store.entity_info else 0}")
            except Exception as e:
                print(f"❌ Error building communities: {e}")
        
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        print(f"✓ Found {len(community_ids)} relevant communities: {community_ids}")
        
        # Step 3: Get community summaries
        print("📚 Step 3: Loading community summaries...")
        community_summaries = self.graph_store.get_community_summaries()
        print(f"🔍 Debug: Total available communities: {len(community_summaries)}")
        print(f"🔍 Debug: Available community IDs: {list(community_summaries.keys())}")
        print(f"🔍 Debug: Looking for community IDs: {community_ids}")
        
        relevant_summaries = {id: summary for id, summary in community_summaries.items() if id in community_ids}
        print(f"✓ Processing {len(relevant_summaries)} community summaries")
        
        if len(relevant_summaries) == 0 and len(community_ids) > 0:
            print("⚠️ Warning: Found community IDs but no matching summaries!")
            print("🔧 Attempting to build communities...")
            try:
                self.graph_store.build_communities()
                community_summaries = self.graph_store.get_community_summaries()
                relevant_summaries = {id: summary for id, summary in community_summaries.items() if id in community_ids}
                print(f"✓ After building: Processing {len(relevant_summaries)} community summaries")
            except Exception as e:
                print(f"❌ Error building communities: {e}")
        
        # Step 3.5: Get relevant chunks from nodes (NEW!)
        print("📄 Step 3.5: Retrieving relevant text chunks...")
        relevant_chunks = self.get_relevant_chunks(query_str, entities)
        print(f"✓ Found {len(relevant_chunks)} relevant text chunks")
        
        # Step 4: Generate answers from each community
        print("🤖 Step 4: Generating answers from communities...")
        community_answers = []
        for i, (id, community_summary) in enumerate(relevant_summaries.items()):
            print(f"  Processing community {id} ({i+1}/{len(relevant_summaries)})...")
            answer = self.generate_answer_from_summary(community_summary, query_str)
            community_answers.append(answer)
            print(f"  ✓ Generated answer for community {id}")

        # Step 4.5: Generate answers from relevant chunks (NEW!)
        print("📝 Step 4.5: Generating answers from text chunks...")
        chunk_answers = []
        for i, chunk in enumerate(relevant_chunks[:3]):  # Limit to top 3 chunks
            print(f"  Processing chunk {i+1}/{min(3, len(relevant_chunks))}...")
            answer = self.generate_answer_from_chunk(chunk, query_str)
            chunk_answers.append(answer)
            print(f"  ✓ Generated answer from chunk {i+1}")

        # Step 5: Aggregate final answer
        print("🔗 Step 5: Aggregating final answer...")
        all_answers = community_answers + chunk_answers
        final_answer = self.aggregate_answers(all_answers)
        print("✅ Query processing completed!")
        
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        """Extract entities relevant to the query by searching the graph store directly."""
        entities = set()
        
        try:
            print(f"🔍 Debug: Trying retriever approach...")
            # First try the original retriever approach
            nodes_retrieved = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            ).retrieve(query_str)
            print(f"🔍 Debug: Retrieved {len(nodes_retrieved)} nodes from index")

            # Try to extract entities from node text using the original pattern
            pattern = (
                r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
            )

            for i, node in enumerate(nodes_retrieved):
                print(f"🔍 Debug: Node {i} text: {node.text[:100]}...")
                matches = re.findall(
                    pattern, node.text, re.MULTILINE | re.IGNORECASE
                )
                print(f"🔍 Debug: Found {len(matches)} matches in node {i}")

                for match in matches:
                    subject = match[0]
                    obj = match[2]
                    entities.add(subject)
                    entities.add(obj)
            
            print(f"🔍 Debug: Entities from retriever: {list(entities)[:5]}{'...' if len(entities) > 5 else ''}")
            
            # If no entities found from retriever, search Neo4j directly
            if not entities:
                print(f"🔍 Debug: No entities from retriever, trying direct Neo4j search...")
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
            
            print(f"🔍 Debug: Query words after filtering: {query_words}")
            
            if not query_words:
                print("⚠️ Warning: No meaningful query words found")
                return entities
            
            # Search for entities whose names or descriptions contain query terms
            for word in query_words:
                print(f"🔍 Debug: Searching for word: '{word}'")
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
                
                word_entities = []
                for record in result:
                    entities.add(record['entity_name'])
                    word_entities.append(record['entity_name'])
                
                print(f"🔍 Debug: Found {len(word_entities)} entities for word '{word}': {word_entities[:3]}{'...' if len(word_entities) > 3 else ''}")
            
            print(f"🔍 Debug: Total entities from direct search: {len(entities)}")
            
            # Also search for entities connected to entities that match query terms
            if entities:
                entity_list = list(entities)[:5]  # Limit to avoid too large queries
                print(f"🔍 Debug: Searching for connected entities to: {entity_list}")
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
                
                connected_entities = []
                for record in result:
                    entities.add(record['entity1'])
                    entities.add(record['entity2'])
                    connected_entities.extend([record['entity1'], record['entity2']])
                
                print(f"🔍 Debug: Found {len(set(connected_entities))} connected entities")
                    
        except Exception as e:
            print(f"Error searching entities in graph: {e}")
            import traceback
            traceback.print_exc()
            
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
        found_entities = []
        missing_entities = []

        if not entity_info:
            print("⚠️ Warning: entity_info is None or empty")
            return community_ids

        for entity in entities:
            if entity in entity_info:
                entity_communities = entity_info[entity]
                community_ids.extend(entity_communities)
                found_entities.append(f"{entity}→{entity_communities}")
            else:
                missing_entities.append(entity)

        print(f"🔍 Debug: Found entities in communities: {found_entities[:3]}{'...' if len(found_entities) > 3 else ''}")
        print(f"🔍 Debug: Missing entities: {missing_entities[:3]}{'...' if len(missing_entities) > 3 else ''}")
        
        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        print(f"    🤖 Generating answer from community summary (length: {len(community_summary)} chars)...")
        
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
        
        print(f"    📤 Sending request to LLM...")
        response = self.llm.chat(messages)
        print(f"    📥 Received response from LLM")
        
        time.sleep(0.5)  # Reduced sleep time for better performance
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print(f"    ✓ Generated answer (length: {len(cleaned_response)} chars)")
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        print(f"🔗 Aggregating {len(community_answers)} community answers...")
        
        if not community_answers:
            print("⚠️ No community answers to aggregate")
            return "I couldn't find relevant information to answer your question."
        
        if len(community_answers) == 1:
            print("ℹ️ Only one community answer, returning directly")
            return community_answers[0]
        
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        
        print(f"📤 Sending aggregation request to LLM...")
        final_response = self.llm.chat(messages)
        print(f"📥 Received final response from LLM")
        
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        
        print(f"✓ Final response generated (length: {len(cleaned_final_response)} chars)")
        return cleaned_final_response

    def get_relevant_chunks(self, query_str, entities):
        """Get relevant text chunks from nodes based on query and entities."""
        if not self.nodes:
            print("⚠️ No nodes available for chunk retrieval")
            return []
        
        relevant_chunks = []
        query_keywords = query_str.lower().split()
        entity_names = [entity.lower() for entity in entities]
        
        print(f"🔍 Searching through {len(self.nodes)} nodes...")
        print(f"🔍 Query keywords: {query_keywords[:5]}{'...' if len(query_keywords) > 5 else ''}")
        print(f"🔍 Entity names: {entity_names[:5]}{'...' if len(entity_names) > 5 else ''}")
        
        for i, node in enumerate(self.nodes):
            node_text = node.text.lower()
            score = 0
            
            # Score based on query keywords
            for keyword in query_keywords:
                if len(keyword) > 2 and keyword in node_text:
                    score += 1
            
            # Score based on entities
            for entity in entity_names:
                if entity in node_text:
                    score += 2  # Entities get higher weight
            
            if score > 0:
                relevant_chunks.append({
                    'node': node,
                    'text': node.text,
                    'score': score,
                    'metadata': node.metadata
                })
        
        # Sort by score descending
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"🔍 Found {len(relevant_chunks)} chunks with relevance scores")
        if relevant_chunks:
            print(f"🔍 Top scores: {[chunk['score'] for chunk in relevant_chunks[:5]]}")
        
        return relevant_chunks

    def generate_answer_from_chunk(self, chunk_data, query):
        """Generate an answer from a text chunk based on a given query using LLM."""
        chunk_text = chunk_data['text']
        score = chunk_data['score']
        
        print(f"    🤖 Generating answer from chunk (score: {score}, length: {len(chunk_text)} chars)...")
        
        prompt = (
            f"Based on the following text content: {chunk_text}, "
            f"how would you answer this query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above text content.",
            ),
        ]
        
        print(f"    📤 Sending chunk request to LLM...")
        response = self.llm.chat(messages)
        print(f"    📥 Received chunk response from LLM")
        
        time.sleep(0.5)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print(f"    ✓ Generated chunk answer (length: {len(cleaned_response)} chars)")
        return cleaned_response

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