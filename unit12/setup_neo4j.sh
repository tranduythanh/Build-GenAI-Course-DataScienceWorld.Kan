#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Start Neo4j
start_neo4j() {
    print_header "Starting Neo4j Docker Container"
    
    if docker-compose ps | grep -q "neo4j-graphrag.*Up"; then
        print_warning "Neo4j is already running!"
        return
    fi
    
    print_status "Starting Neo4j container..."
    docker-compose up -d
    
    print_status "Waiting for Neo4j to be ready..."
    sleep 30
    
    # Wait for Neo4j to be healthy
    print_status "Checking Neo4j health..."
    for i in {1..30}; do
        if curl -s http://localhost:7474 > /dev/null; then
            print_status "Neo4j is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Neo4j failed to start properly. Check logs with: docker-compose logs neo4j"
            exit 1
        fi
        sleep 2
    done
}

# Stop Neo4j
stop_neo4j() {
    print_header "Stopping Neo4j Docker Container"
    docker-compose down
    print_status "Neo4j stopped."
}

# Show Neo4j status
status_neo4j() {
    print_header "Neo4j Status"
    docker-compose ps
    
    if docker-compose ps | grep -q "neo4j-graphrag.*Up"; then
        print_status "Neo4j is running!"
        print_status "Web interface: http://localhost:7474"
        print_status "Bolt connection: bolt://localhost:7687"
        print_status "Username: neo4j"
        print_status "Password: password"
    else
        print_warning "Neo4j is not running."
    fi
}

# Show logs
logs_neo4j() {
    print_header "Neo4j Logs"
    docker-compose logs -f neo4j
}

# Reset Neo4j (remove all data)
reset_neo4j() {
    print_header "Resetting Neo4j Database"
    print_warning "This will delete ALL data in the Neo4j database!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping Neo4j..."
        docker-compose down
        print_status "Removing data volumes..."
        docker-compose down -v
        print_status "Neo4j database reset complete."
    else
        print_status "Reset cancelled."
    fi
}

# Show help
show_help() {
    echo "Neo4j Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start    Start Neo4j container"
    echo "  stop     Stop Neo4j container"
    echo "  status   Show Neo4j status"
    echo "  logs     Show Neo4j logs (follow mode)"
    echo "  reset    Reset Neo4j database (removes all data)"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start Neo4j"
    echo "  $0 status    # Check if Neo4j is running"
    echo "  $0 logs      # View logs"
}

# Main script
main() {
    check_docker
    
    case "${1:-help}" in
        start)
            start_neo4j
            status_neo4j
            ;;
        stop)
            stop_neo4j
            ;;
        status)
            status_neo4j
            ;;
        logs)
            logs_neo4j
            ;;
        reset)
            reset_neo4j
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@" 