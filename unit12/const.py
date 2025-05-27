import re
import os
from typing import Optional

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default value."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is required but not set")
    return value

# Regular expression patterns for parsing LLM responses
entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

# Knowledge Graph extraction template
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

# Neo4J Database Configuration (Local Docker)
NEO4J_URI = get_env_var("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = get_env_var("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_env_var("NEO4J_PASSWORD", "password")

# API Configuration
TOGETHER_API_KEY = get_env_var("TOGETHER_API_KEY", "524a0fda9f6199191bb6252d8c9d0f07edfb630ecaf91a2dbe76c95de4749d15")

# Model Configuration
DEFAULT_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Data Configuration
DATA_URL = "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_MAX_PATHS_PER_CHUNK = 2
DEFAULT_SIMILARITY_TOP_K = 10

# GraphRAG Configuration
DEFAULT_MAX_CLUSTER_SIZE = 5
DEFAULT_NUM_WORKERS = 4