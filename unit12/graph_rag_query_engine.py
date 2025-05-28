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
        print(f"\tDebug: Entity info available: {self.graph_store.entity_info is not None}")
        print(f"\tDebug: Entity info type: {type(self.graph_store.entity_info)}")
        print(f"\tDebug: GraphRAGStore object ID: {id(self.graph_store)}")
        
        if hasattr(self.graph_store, 'entity_info') and self.graph_store.entity_info:
            print(f"\tDebug: Entity info size: {len(self.graph_store.entity_info)}")
            print(f"\tDebug: Sample entities in info: {list(self.graph_store.entity_info.keys())[:5]}")
        else:
            print("⚠️ Warning: No entity_info found, attempting to build communities...")
            print(f"\tDebug: Forcing load from cache first...")
            # Try to force load from cache again
            try:
                load_success = self.graph_store.load_community_summaries_from_file()
                print(f"\tDebug: Force load result: {load_success}")
                if load_success:
                    print(f"\tDebug: After force load - entity_info: {len(self.graph_store.entity_info) if self.graph_store.entity_info else 0} entities")
            except Exception as e:
                print(f"\tDebug: Force load failed: {e}")
            
            # If still no entity_info, build communities
            if not self.graph_store.entity_info:
                print("\tDebug: Still no entity_info, building communities...")
                try:
                    self.graph_store.build_communities()
                    print(f"✓ Communities built, entity_info size: {len(self.graph_store.entity_info) if self.graph_store.entity_info else 0}")
                except Exception as e:
                    print(f"❌ Error building communities: {e}")
            else:
                print(f"\tDebug: Force load successful, entity_info size: {len(self.graph_store.entity_info)}")
        
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        print(f"✓ Found {len(community_ids)} relevant communities: {community_ids}")
        
        # Step 3: Get community summaries
        print("📚 Step 3: Loading community summaries...")
        community_summaries = self.graph_store.get_community_summaries()
        print(f"\tDebug: Total available communities: {len(community_summaries)}")
        print(f"\tDebug: Available community IDs: {list(community_summaries.keys())}")
        print(f"\tDebug: Looking for community IDs: {community_ids}")
        
        # Convert community_ids to strings to match the keys in community_summaries
        community_ids_str = [str(id) for id in community_ids]
        print(f"\tDebug: Community IDs as strings: {community_ids_str}")
        
        relevant_summaries = {id: summary for id, summary in community_summaries.items() if id in community_ids_str}
        print(f"✓ Processing {len(relevant_summaries)} community summaries")
        
        if len(relevant_summaries) == 0 and len(community_ids) > 0:
            print("⚠️ Warning: Found community IDs but no matching summaries!")
            print("🔧 Attempting to build communities...")
            try:
                self.graph_store.build_communities()
                community_summaries = self.graph_store.get_community_summaries()
                relevant_summaries = {id: summary for id, summary in community_summaries.items() if id in community_ids_str}
                print(f"✓ After building: Processing {len(relevant_summaries)} community summaries")
            except Exception as e:
                print(f"❌ Error building communities: {e}")
        
        # Step 3.5: Get relevant chunks from nodes (NEW!)
        print("📄 Step 3.5: Retrieving relevant text chunks...")
        relevant_chunks = self.get_relevant_chunks(query_str, entities)
        print(f"✓ Found {len(relevant_chunks)} relevant text chunks")
        
        # Step 4: Generate answers from communities (OPTIMIZED - batch processing)
        print("🤖 Step 4: Generating answers from communities...")
        
        # Limit to top 10 best communities based on relevance
        if len(relevant_summaries) > 10:
            print(f"  ⚡ Limiting from {len(relevant_summaries)} to top 10 communities for efficiency")
            # Convert to list of tuples for sorting
            summary_items = list(relevant_summaries.items())
            # Sort by summary length (longer summaries likely more relevant) 
            # and take top 10
            summary_items.sort(key=lambda x: len(x[1]), reverse=True)
            relevant_summaries = dict(summary_items[:10])
        
        community_answers = []
        if relevant_summaries:
            # Batch all community summaries into a single LLM call
            print(f"  📦 Processing {len(relevant_summaries)} communities in a single batch call...")
            
            # Prepare batch prompt with all community summaries
            batch_prompt_parts = []
            for i, (community_id, summary) in enumerate(relevant_summaries.items(), 1):
                batch_prompt_parts.append(f"Community {community_id}:\n{summary}")
            
            batch_summaries = "\n\n---\n\n".join(batch_prompt_parts)
            
            print(f"  📤 Sending batch request to LLM for {len(relevant_summaries)} communities...")
            batch_answer = self.generate_batch_answer_from_summaries(batch_summaries, query_str)
            print(f"  📥 Received batch response from LLM")
            print(f"  ✓ Generated batch answer (length: {len(batch_answer)} chars)")
            
            community_answers = [batch_answer]
        else:
            print("  ⚠️ No relevant communities found")

        # Step 4.5: Generate answers from relevant chunks (limit to 3)
        print("📝 Step 4.5: Generating answers from text chunks...")
        chunk_answers = []
        
        # Limit to top 3 chunks and batch process them
        top_chunks = relevant_chunks[:3]
        if top_chunks:
            print(f"  📦 Processing {len(top_chunks)} chunks in a single batch call...")
            
            # Prepare batch prompt with all chunks
            batch_chunk_parts = []
            for i, chunk in enumerate(top_chunks, 1):
                chunk_text = chunk['text']
                score = chunk['score']
                batch_chunk_parts.append(f"Text Chunk {i} (Relevance Score: {score}):\n{chunk_text}")
            
            batch_chunks = "\n\n---\n\n".join(batch_chunk_parts)
            
            print(f"  📤 Sending batch chunk request to LLM for {len(top_chunks)} chunks...")
            batch_chunk_answer = self.generate_batch_answer_from_chunks(batch_chunks, query_str)
            print(f"  📥 Received batch chunk response from LLM")
            print(f"  ✓ Generated batch chunk answer (length: {len(batch_chunk_answer)} chars)")
            
            chunk_answers = [batch_chunk_answer]
        else:
            print("  ⚠️ No relevant chunks found")

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
            print(f"\tDebug: Trying retriever approach...")
            # First try the original retriever approach
            nodes_retrieved = self.index.as_retriever(
                similarity_top_k=similarity_top_k
            ).retrieve(query_str)
            print(f"\tDebug: Retrieved {len(nodes_retrieved)} nodes from index")

            # Try to extract entities from node text using the original pattern
            pattern = (
                r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
            )

            for i, node in enumerate(nodes_retrieved):
                print(f"\tDebug: Node {i} text: {node.text[:100]}...")
                matches = re.findall(
                    pattern, node.text, re.MULTILINE | re.IGNORECASE
                )
                print(f"\tDebug: Found {len(matches)} matches in node {i}")

                for match in matches:
                    subject = match[0]
                    obj = match[2]
                    entities.add(subject)
                    entities.add(obj)
            
            print(f"\tDebug: Entities from retriever: {list(entities)[:5]}{'...' if len(entities) > 5 else ''}")
            
            # If no entities found from retriever, search Neo4j directly
            if not entities:
                print(f"\tDebug: No entities from retriever, trying direct Neo4j search...")
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
            
            print(f"\tDebug: Query words after filtering: {query_words}")
            
            if not query_words:
                print("⚠️ Warning: No meaningful query words found")
                return entities
            
            # Search for entities whose names or descriptions contain query terms
            for word in query_words:
                print(f"\tDebug: Searching for word: '{word}'")
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
                
                print(f"\tDebug: Found {len(word_entities)} entities for word '{word}': {word_entities[:3]}{'...' if len(word_entities) > 3 else ''}")
            
            print(f"\tDebug: Total entities from direct search: {len(entities)}")
            
            # Also search for entities connected to entities that match query terms
            if entities:
                entity_list = list(entities)[:5]  # Limit to avoid too large queries
                print(f"\tDebug: Searching for connected entities to: {entity_list}")
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
                
                print(f"\tDebug: Found {len(set(connected_entities))} connected entities")
                    
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

        print(f"\tDebug: Found entities in communities: {found_entities[:3]}{'...' if len(found_entities) > 3 else ''}")
        print(f"\tDebug: Missing entities: {missing_entities[:3]}{'...' if len(missing_entities) > 3 else ''}")
        
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

    def generate_batch_answer_from_summaries(self, batch_summaries, query):
        """Generate an answer from multiple community summaries in a single LLM call."""
        print(f"    🤖 Generating batch answer from {batch_summaries.count('Community')} community summaries...")
        
        prompt = (
            f"You are provided with multiple community summaries from a knowledge graph. "
            f"Please analyze all the information and provide a comprehensive answer to the query.\n\n"
            f"Community Summaries:\n{batch_summaries}\n\n"
            f"Query: {query}\n\n"
            f"Please provide a well-structured answer that synthesizes information from all relevant communities."
        )
        
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user", 
                content="Please analyze all the community information and provide a comprehensive answer to the query."
            ),
        ]
        
        print(f"    📤 Sending batch request to LLM...")
        response = self.llm.chat(messages)
        print(f"    📥 Received batch response from LLM")
        
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print(f"    ✓ Generated batch answer (length: {len(cleaned_response)} chars)")
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

    def generate_batch_answer_from_chunks(self, batch_chunks, query):
        """Generate an answer from multiple text chunks in a single LLM call."""
        print(f"    🤖 Generating batch answer from {batch_chunks.count('Text Chunk')} text chunks...")
        
        prompt = (
            f"You are provided with multiple text chunks from a knowledge graph. "
            f"Please analyze all the information and provide a comprehensive answer to the query.\n\n"
            f"Text Chunks:\n{batch_chunks}\n\n"
            f"Query: {query}\n\n"
            f"Please provide a well-structured answer that synthesizes information from all relevant text chunks."
        )
        
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user", 
                content="Please analyze all the text chunk information and provide a comprehensive answer to the query."
            ),
        ]
        
        print(f"    📤 Sending batch request to LLM...")
        response = self.llm.chat(messages)
        print(f"    📥 Received batch response from LLM")
        
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print(f"    ✓ Generated batch answer (length: {len(cleaned_response)} chars)")
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