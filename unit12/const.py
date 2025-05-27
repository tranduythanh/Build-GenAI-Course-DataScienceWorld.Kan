import re

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

# Neo4J Database Configuration
NEO4J_URI = "neo4j+s://6b70473e.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "fPpNzCHUnmtxRSjpLJRzeufhFjb5xNpTG4JdSfRgd9M"

# API Configuration
TOGETHER_API_KEY = "524a0fda9f6199191bb6252d8c9d0f07edfb630ecaf91a2dbe76c95de4749d15"

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