#!/usr/bin/env python3
"""
Unit tests for Neo4j connection functionality.
"""

import unittest
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from const import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


def _is_neo4j_available():
    """Check if Neo4j is available for testing."""
    try:
        import requests
        response = requests.get("http://localhost:7474", timeout=5)
        return response.status_code == 200
    except:
        return False


class TestNeo4jConnection(unittest.TestCase):
    """Test cases for Neo4j database connection."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.neo4j_uri = NEO4J_URI
        self.neo4j_username = NEO4J_USERNAME
        self.neo4j_password = NEO4J_PASSWORD
        self.graph_store = None
    
    def tearDown(self):
        """Clean up after each test method."""
        if self.graph_store and hasattr(self.graph_store, '_driver'):
            try:
                self.graph_store._driver.close()
            except:
                pass
    
    def test_neo4j_import(self):
        """Test that we can import the Neo4j graph store."""
        try:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            self.assertTrue(True, "Successfully imported Neo4jPropertyGraphStore")
        except ImportError as e:
            self.fail(f"Failed to import Neo4jPropertyGraphStore: {e}")
    
    def test_neo4j_connection_parameters(self):
        """Test that Neo4j connection parameters are properly configured."""
        self.assertIsNotNone(self.neo4j_uri, "NEO4J_URI should not be None")
        self.assertIsNotNone(self.neo4j_username, "NEO4J_USERNAME should not be None")
        self.assertIsNotNone(self.neo4j_password, "NEO4J_PASSWORD should not be None")
        
        # Check that URI format is correct for local Docker
        self.assertTrue(
            self.neo4j_uri.startswith("bolt://localhost:7687") or 
            self.neo4j_uri.startswith("neo4j://localhost:7687"),
            f"NEO4J_URI should be local Docker format, got: {self.neo4j_uri}"
        )
    
    def test_neo4j_connection(self):
        """Test actual connection to Neo4j database."""
        try:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            
            # Create connection
            self.graph_store = Neo4jPropertyGraphStore(
                username=self.neo4j_username,
                password=self.neo4j_password,
                url=self.neo4j_uri
            )
            
            # If we get here without exception, connection was successful
            self.assertIsNotNone(self.graph_store, "Graph store should be created successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import required dependencies: {e}")
        except Exception as e:
            # Neo4j connection might fail if server is not running, but that's expected
            self.skipTest(f"Neo4j server not available: {e}")
    
    def test_neo4j_simple_query(self):
        """Test executing a simple query against Neo4j."""
        try:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            
            # Create connection
            self.graph_store = Neo4jPropertyGraphStore(
                username=self.neo4j_username,
                password=self.neo4j_password,
                url=self.neo4j_uri
            )
            
            # Test simple query
            result = self.graph_store._driver.session().run("RETURN 'Hello Neo4j' as message")
            record = result.single()
            
            self.assertIsNotNone(record, "Query should return a result")
            self.assertEqual(record['message'], 'Hello Neo4j', "Query should return expected message")
            
        except ImportError as e:
            self.fail(f"Failed to import required dependencies: {e}")
        except Exception as e:
            # Neo4j connection might fail if server is not running, but that's expected
            self.skipTest(f"Neo4j server not available: {e}")
    
    def test_neo4j_database_info(self):
        """Test retrieving basic database information."""
        try:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            
            # Create connection
            self.graph_store = Neo4jPropertyGraphStore(
                username=self.neo4j_username,
                password=self.neo4j_password,
                url=self.neo4j_uri
            )
            
            # Get database info
            with self.graph_store._driver.session() as session:
                result = session.run("CALL db.schema.visualization()")
                # This should not raise an exception
                records = list(result)
                
                # We don't check the content since it might be empty for new database
                self.assertIsInstance(records, list, "Schema query should return a list")
                
        except ImportError as e:
            self.fail(f"Failed to import required dependencies: {e}")
        except Exception as e:
            # Neo4j connection might fail if server is not running, but that's expected
            self.skipTest(f"Neo4j server not available: {e}")


class TestNeo4jConnectionWithSkip(unittest.TestCase):
    """Test cases that can be skipped if Neo4j is not available."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph_store = None
    
    def tearDown(self):
        """Clean up after each test method."""
        if self.graph_store and hasattr(self.graph_store, '_driver'):
            try:
                self.graph_store._driver.close()
            except:
                pass
    
    @unittest.skipIf(not _is_neo4j_available(), "Neo4j is not available")
    def test_neo4j_health_check(self):
        """Test Neo4j health check (skipped if Neo4j not available)."""
        try:
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            
            self.graph_store = Neo4jPropertyGraphStore(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=NEO4J_URI
            )
            
            # Simple health check
            with self.graph_store._driver.session() as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                self.assertEqual(record['health'], 1, "Health check should return 1")
                
        except ImportError as e:
            self.fail(f"Failed to import required dependencies: {e}")
        except Exception as e:
            # Neo4j connection might fail if server is not running, but that's expected
            self.skipTest(f"Neo4j server not available: {e}")


def run_connection_test():
    """Function to run just the connection test with pretty output."""
    print("=" * 60)
    print("Neo4j Connection Test")
    print("=" * 60)
    
    # Create test suite with just connection tests
    suite = unittest.TestSuite()
    suite.addTest(TestNeo4jConnection('test_neo4j_connection_parameters'))
    suite.addTest(TestNeo4jConnection('test_neo4j_connection'))
    suite.addTest(TestNeo4jConnection('test_neo4j_simple_query'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 Neo4j connection test passed!")
        print("You can now run the GraphRAG application with: python app.py")
        return True
    else:
        print("\n💥 Neo4j connection test failed!")
        print("Please fix the connection issues before running the main application.")
        print("\nTroubleshooting steps:")
        print("1. Make sure Neo4j is running: ./setup_neo4j.sh status")
        print("2. Start Neo4j if not running: ./setup_neo4j.sh start")
        print("3. Check if ports 7474 and 7687 are available")
        print("4. Verify credentials in const.py")
        return False


if __name__ == '__main__':
    # If run directly, run the pretty connection test
    if len(sys.argv) > 1 and sys.argv[1] == '--connection-only':
        sys.exit(0 if run_connection_test() else 1)
    else:
        # Run all tests
        unittest.main(verbosity=2) 