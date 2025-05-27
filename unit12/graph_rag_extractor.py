import asyncio
import csv
import os
import nest_asyncio
from io import StringIO
from typing import Any, List, Callable, Optional, Union

nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode, TextNode
from llama_index.llms.openai import OpenAI
from const import (
    OPENAI_API_KEY, 
    DEFAULT_MODEL, 
    CSV_KG_EXTRACT_TMPL,
    DEFAULT_MAX_PATHS_PER_CHUNK
)

def parse_csv_section(lines, section_type):
    """Parse CSV lines for either entities or relationships."""
    results = []
    for line in lines:
        if not line or line.startswith('#') or 'entity,name,type' in line or 'relationship,source,target' in line:
            continue
            
        try:
            # Use CSV reader to properly handle quoted fields
            csv_reader = csv.reader(StringIO(line))
            row = next(csv_reader)
            
            if len(row) >= 4 and row[0].lower() == section_type:
                if section_type == "entity":
                    # Format: entity,name,type,description
                    results.append((row[1].strip(), row[2].strip(), row[3].strip()))
                elif section_type == "relationship":
                    # Format: relationship,source,target,relation,description
                    if len(row) >= 5:
                        results.append((row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()))
                    
        except Exception as e:
            print(f"Error parsing CSV line '{line}': {e}")
            continue
            
    return results



def parse_csv_triplets_fn(response_str: str):
    """Parse function for extracting entities and relationships from CSV format."""
    
    
    entities = []
    relationships = []
    
    try:
        # Split response into entities and relationships sections
        lines = response_str.strip().split('\n')
        
        current_section = None
        entity_lines = []
        relationship_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('```'):
                continue
                
            if line.upper() == "ENTITIES:" or "ENTITIES" in line.upper():
                current_section = "entities"
                continue
            elif line.upper() == "RELATIONSHIPS:" or "RELATIONSHIPS" in line.upper():
                current_section = "relationships"
                continue
            elif line.startswith('entity,name,type') or line.startswith('relationship,source,target'):
                # Skip header lines
                continue
            elif current_section == "entities" and line.startswith('entity,'):
                entity_lines.append(line)
            elif current_section == "relationships" and line.startswith('relationship,'):
                relationship_lines.append(line)
        
        # Process entities and relationships
        entities = parse_csv_section(entity_lines, "entity")
        relationships = parse_csv_section(relationship_lines, "relationship")
        
        print(f"✓ Successfully parsed {len(entities)} entities and {len(relationships)} relationships")
        
    except Exception as e:
        print(f"CSV Parse error: {e}")
        
    return entities, relationships



class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_csv_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm") # Return content as llm response rather than structured json format
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError as e:
            print(f"Parse error: {e}")
            entities = []
            entities_relationship = []
        except Exception as e:
            print(f"Extraction error: {e}")
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, []) # pop out existing nodes
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, []) # pop out relations
        entity_metadata = node.metadata.copy() # copy metadata

        # Construction nodes
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()

        # Constructing relationship
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
    
if __name__ == "__main__":
    # Setup OpenAI LLM
    print("Setting up OpenAI LLM...")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = OpenAI(model=DEFAULT_MODEL)
    Settings.llm = llm
    
    # Test LLM connection
    print("Testing LLM connection...")
    test_response = llm.complete("What is 2+2?")
    print(f"LLM Test Response: {test_response}")
    
    # Create sample text nodes with content to extract from
    sample_texts = [
        "Apple Inc. is a technology company founded by Steve Jobs. The company is headquartered in Cupertino, California. Tim Cook is the current CEO of Apple Inc.",
        "Microsoft Corporation was founded by Bill Gates and Paul Allen. The company develops software products including Windows operating system. Satya Nadella serves as the CEO of Microsoft.",
        "Tesla is an electric vehicle manufacturer led by Elon Musk. The company produces electric cars and energy storage systems. Tesla's headquarters is located in Austin, Texas."
    ]
    
    # Create TextNode objects
    nodes = []
    for i, text in enumerate(sample_texts):
        node = TextNode(
            text=text,
            id_=f"node_{i}",
            metadata={"source": f"document_{i}"}
        )
        nodes.append(node)
    
    try:
        # Create the extractor with proper configuration
        print("Creating GraphRAG Extractor with OpenAI LLM and CSV format...")
        extractor = GraphRAGExtractor(
            llm=llm,
            extract_prompt=CSV_KG_EXTRACT_TMPL,
            max_paths_per_chunk=DEFAULT_MAX_PATHS_PER_CHUNK,
            num_workers=2
        )
        
        # Extract triples from the nodes
        print("Extracting knowledge graph from sample texts...")
        result_nodes = extractor(nodes, show_progress=True)
        
        # Display results
        print(f"\nProcessed {len(result_nodes)} nodes:")
        for i, node in enumerate(result_nodes):
            print(f"\n--- Node {i+1} ---")
            print(f"Text: {node.text[:100]}...")
            
            # Display extracted entities
            entities = node.metadata.get(KG_NODES_KEY, [])
            print(f"Entities found: {len(entities)}")
            for entity in entities[:5]:  # Show first 5
                print(f"  - {entity.name} ({entity.label}) - {entity.properties.get('entity_description', 'No description')[:50]}...")
            
            # Display extracted relations
            relations = node.metadata.get(KG_RELATIONS_KEY, [])
            print(f"Relations found: {len(relations)}")
            for relation in relations[:5]:  # Show first 5
                rel_desc = relation.properties.get('relationship_description', 'No description')[:50]
                print(f"  - {relation.source_id} --{relation.label}--> {relation.target_id} ({rel_desc}...)")
        
        print("\nKnowledge graph extraction completed successfully!")
        
        # Summary statistics
        total_entities = sum(len(node.metadata.get(KG_NODES_KEY, [])) for node in result_nodes)
        total_relations = sum(len(node.metadata.get(KG_RELATIONS_KEY, [])) for node in result_nodes)
        print(f"\nSummary:")
        print(f"Total entities extracted: {total_entities}")
        print(f"Total relationships extracted: {total_relations}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()