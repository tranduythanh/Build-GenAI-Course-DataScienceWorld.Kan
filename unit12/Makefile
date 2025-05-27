.PHONY: help install neo4j-start neo4j-stop neo4j-status neo4j-reset test-connection run clean

# Default target
help:
	@echo "GraphRAG Application Commands"
	@echo "============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install Python dependencies"
	@echo "  neo4j-start      Start Neo4j Docker container"
	@echo "  neo4j-stop       Stop Neo4j Docker container"
	@echo "  neo4j-status     Show Neo4j status"
	@echo "  neo4j-reset      Reset Neo4j database (removes all data)"
	@echo ""
	@echo "Application Commands:"
	@echo "  test             Run all unit tests"
	@echo "  test-connection  Test Neo4j connection"
	@echo "  run              Run the GraphRAG application"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean            Clean up Docker containers and volumes"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install && make neo4j-start && make test-connection && make run"

# Install dependencies
install:
	pip install -r requirements.txt

# Neo4j management
neo4j-start:
	./setup_neo4j.sh start

neo4j-stop:
	./setup_neo4j.sh stop

neo4j-status:
	./setup_neo4j.sh status

neo4j-reset:
	./setup_neo4j.sh reset

# Application commands
test:
	cd tests && python -m unittest discover -v

test-connection:
	cd tests && python test_neo4j_connection.py --connection-only

run:
	python app.py

# Cleanup
clean:
	docker-compose down -v
	docker system prune -f

# Complete setup
setup: install neo4j-start test-connection
	@echo "Setup complete! You can now run: make test && make run"

# Quick restart
restart: neo4j-stop neo4j-start test-connection
	@echo "Neo4j restarted successfully!" 