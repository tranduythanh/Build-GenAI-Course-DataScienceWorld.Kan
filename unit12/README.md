# GraphRAG Application

A complete GraphRAG (Graph + Retrieval Augmented Generation) implementation using LlamaIndex and Neo4j, converted from Jupyter notebook to run in terminal environment.

## Overview

This application implements the GraphRAG pipeline that combines the strengths of Retrieval Augmented Generation (RAG) and Query-Focused Summarization (QFS) to effectively handle complex queries over large text datasets.

## Features

- **Knowledge Graph Extraction**: Extracts entities and relationships from text using LLM
- **Community Detection**: Groups related entities using hierarchical Leiden algorithm
- **Graph Storage**: Uses Neo4j for scalable graph storage
- **Semantic Search**: Embedding-based retrieval for relevant information
- **Query Processing**: Multi-step query processing with community-based summarization

## Project Structure

```
├── app.py                      # Main application entry point
├── const.py                    # Configuration constants and templates
├── graph_rag_extractor.py      # Knowledge graph extraction logic
├── graph_rag_store.py          # Neo4j graph store with community detection
├── graph_rag_query_engine.py   # Query processing engine
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- Neo4j database (cloud or local)
- Together API key for LLM access

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Start Neo4j:**
   ```bash
   ./setup_neo4j.sh start
   ```

2. **Test connection:**
   ```bash
   python test_neo4j_connection.py
   ```

3. **Run GraphRAG:**
   ```bash
   python app.py
   ```

4. **Access Neo4j Browser:** http://localhost:7474 (neo4j/password)

### Alternative: Using Makefile

For convenience, you can use the provided Makefile:

```bash
# See all available commands
make help

# Complete setup
make setup

# Or step by step
make install          # Install dependencies
make neo4j-start      # Start Neo4j
make test-connection  # Test connection
make run              # Run application
```

## Configuration

Update the constants in `const.py`:

- **Neo4j Database**: Update `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- **API Keys**: Update `TOGETHER_API_KEY`
- **Model Settings**: Modify `DEFAULT_MODEL` if needed

## Usage

### Prerequisites

1. **Start Neo4j Docker container first:**
   ```bash
   ./setup_neo4j.sh start
   ```

2. **Verify Neo4j is running:**
   ```bash
   ./setup_neo4j.sh status
   ```

3. **Test Neo4j connection:**
   ```bash
   python test_neo4j_connection.py
   ```

### Basic Usage

Run the complete GraphRAG pipeline:

```bash
python app.py
```

This will:
1. Load news articles dataset (50 samples by default)
2. Create text chunks and extract knowledge graphs
3. Build communities and generate summaries
4. Run sample queries against the graph

### Custom Queries

Modify the `run_queries()` function in `app.py` to add your own queries:

```python
queries = [
    "Your custom query here",
    "Another query about specific topics"
]
```

### Adjusting Parameters

You can modify various parameters in `const.py`:

- `DEFAULT_CHUNK_SIZE`: Size of text chunks (default: 1024)
- `DEFAULT_MAX_PATHS_PER_CHUNK`: Max relationships per chunk (default: 2)
- `DEFAULT_SIMILARITY_TOP_K`: Number of similar chunks to retrieve (default: 10)
- `DEFAULT_MAX_CLUSTER_SIZE`: Maximum community size (default: 5)

## Components

### GraphRAGExtractor
- Extracts entities and relationships from text
- Uses custom prompts for structured extraction
- Supports async processing for better performance

### GraphRAGStore
- Extends Neo4j PropertyGraphStore
- Implements community detection using hierarchical Leiden
- Generates community summaries using LLM

### GraphRAGQueryEngine
- Processes queries using community summaries
- Retrieves relevant entities through embedding similarity
- Aggregates answers from multiple communities

## Neo4j Setup

### Local Setup (Docker) - Recommended

We've simplified Neo4j setup with Docker Compose. Choose one of the following methods:

#### Option 1: Using the Setup Script (Recommended)
```bash
# Make script executable (if not already)
chmod +x setup_neo4j.sh

# Start Neo4j
./setup_neo4j.sh start

# Check status
./setup_neo4j.sh status

# View logs
./setup_neo4j.sh logs

# Stop Neo4j
./setup_neo4j.sh stop
```

#### Option 2: Using Docker Compose Directly
```bash
# Start Neo4j
docker-compose up -d

# Stop Neo4j
docker-compose down

# View logs
docker-compose logs -f neo4j
```

#### Option 3: Manual Docker Command
```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v neo4j_data:/data \
    -v neo4j_logs:/logs \
    --name neo4j-graphrag \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_dbms_memory_heap_max__size=2G \
    -d neo4j:5.15-community
```

### Accessing Neo4j

Once Neo4j is running:
- **Web Interface**: http://localhost:7474
- **Bolt Connection**: bolt://localhost:7687
- **Username**: neo4j
- **Password**: password

### Configuration

The application is now configured to use local Neo4j Docker by default. The connection settings in `const.py` are:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j" 
NEO4J_PASSWORD = "password"
```

### Cloud Setup (Optional)
If you prefer using Neo4j AuraDB cloud instance, update the credentials in `const.py`:

```python
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-password"
```

## Data Source

The application uses a news articles dataset from GitHub:
- Source: https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv
- Contains ~2,500 news articles with title and text
- Uses first 50 articles by default for testing

## Troubleshooting

### Common Issues

1. **Neo4j Connection Error**: Verify database credentials and network connectivity
2. **API Rate Limits**: The application includes sleep delays to respect API limits
3. **Memory Issues**: Reduce number of nodes processed or chunk size
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Tips

- Start with fewer documents (10-20) for testing
- Increase `max_paths_per_chunk` for more detailed extraction
- Adjust `similarity_top_k` based on query complexity

## Understanding the Leiden Algorithm

### What is the Leiden Algorithm?

The **Leiden Algorithm** is a community detection algorithm developed by researchers at Leiden University that improves upon the popular Louvain algorithm. It's used in this GraphRAG implementation to find well-connected communities within the knowledge graph.

### Key Features

- **Guarantees well-connected communities**: Unlike Louvain, Leiden ensures communities are never disconnected
- **Better quality**: Finds higher-quality community structures
- **Faster performance**: Often runs faster than Louvain, especially on larger networks
- **Hierarchical clustering**: Recursively merges communities by optimizing modularity

### Why Leiden over Louvain?

The Louvain algorithm, while popular, has a significant flaw: it can produce arbitrarily badly connected communities that may even be disconnected. The Leiden algorithm fixes this by:

1. **Local moving of nodes**: Efficiently moves nodes between communities
2. **Refinement phase**: Ensures communities remain well-connected
3. **Aggregation**: Creates hierarchical community structures

### Educational Resources

#### Research Papers
- **Original Paper**: [From Louvain to Leiden: guaranteeing well-connected communities](https://www.nature.com/articles/s41598-019-41695-z) - Nature Scientific Reports (2019)
- **ArXiv Version**: [From Louvain to Leiden](https://arxiv.org/abs/1810.08473)

#### Algorithm Documentation
- **Neo4j Implementation**: [Leiden Algorithm - Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
- **Python Implementation**: [leidenalg GitHub Repository](https://github.com/vtraag/leidenalg)
- **Memgraph Documentation**: [Leiden Community Detection](https://memgraph.com/docs/advanced-algorithms/available-algorithms/leiden_community_detection)

#### Educational Articles
- **CWTS Blog**: [Using the Leiden algorithm to find well-connected clusters](https://www.cwts.nl/blog?article=n-r2u2a4)
- **Wikipedia**: [Leiden Algorithm](https://en.wikipedia.org/wiki/Leiden_algorithm)

### Algorithm Parameters in Our Implementation

In `const.py`, you can adjust these Leiden-related parameters:

- `DEFAULT_MAX_CLUSTER_SIZE`: Maximum community size (default: 5)
- `DEFAULT_NUM_WORKERS`: Number of parallel workers (default: 4)

### How It Works in GraphRAG

1. **Knowledge Graph Construction**: Entities and relationships are extracted from text
2. **Community Detection**: Leiden algorithm groups related entities into communities
3. **Summarization**: Each community gets an AI-generated summary
4. **Query Processing**: Questions are answered using relevant community summaries

### Performance Benefits

- **Speed**: Up to 20x faster than Louvain on large networks
- **Quality**: Better modularity scores and community structure
- **Reliability**: Guaranteed connected communities
- **Scalability**: Works efficiently on networks with millions of nodes

### Further Learning

For deeper understanding of community detection and graph algorithms:

- **Graph Theory**: Study basic graph concepts and community structure
- **Network Science**: Learn about complex networks and their properties  
- **Machine Learning**: Understand clustering and unsupervised learning methods
- **Information Retrieval**: Explore how community detection aids search and recommendation

## License

This project is based on the LlamaIndex GraphRAG cookbook and is provided as-is for educational and research purposes.


