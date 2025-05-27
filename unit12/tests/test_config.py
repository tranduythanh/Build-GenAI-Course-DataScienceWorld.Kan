#!/usr/bin/env python3
"""
Unit tests for configuration settings.
"""

import unittest
import sys
import os
import re

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from const import CSV_KG_EXTRACT_TMPL
from graph_rag_extractor import parse_csv_triplets_fn


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_csv_prompt_template(self):
        """Test that CSV prompt template is valid."""
        self.assertIsNotNone(CSV_KG_EXTRACT_TMPL, "CSV_KG_EXTRACT_TMPL should be defined")
        self.assertIn("{text}", CSV_KG_EXTRACT_TMPL, "Template should contain {text} placeholder")
        self.assertIn("{max_knowledge_triplets}", CSV_KG_EXTRACT_TMPL, "Template should contain {max_knowledge_triplets} placeholder")
    
    def test_csv_parsing_functionality(self):
        """Test CSV parsing functionality."""
        test_csv_response = """
        ENTITIES:
        entity,Apple Inc,Organization,Apple Inc. is a technology company
        entity,Steve Jobs,Person,Steve Jobs was the co-founder of Apple

        RELATIONSHIPS:
        relationship,Apple Inc,Steve Jobs,founded_by,Steve Jobs co-founded Apple Inc.
        """
        
        entities, relationships = parse_csv_triplets_fn(test_csv_response)
        
        self.assertEqual(len(entities), 2, "Should find exactly two entities")
        self.assertEqual(len(relationships), 1, "Should find exactly one relationship")
        
        # Test entity parsing
        apple_entity = entities[0]
        self.assertEqual(apple_entity[0], "Apple Inc", "Entity name should be extracted correctly")
        self.assertEqual(apple_entity[1], "Organization", "Entity type should be extracted correctly")
        self.assertEqual(apple_entity[2], "Apple Inc. is a technology company", "Entity description should be extracted correctly")
        
        # Test relationship parsing
        rel = relationships[0]
        self.assertEqual(rel[0], "Apple Inc", "Source entity should be extracted correctly")
        self.assertEqual(rel[1], "Steve Jobs", "Target entity should be extracted correctly")
        self.assertEqual(rel[2], "founded_by", "Relation should be extracted correctly")
        self.assertEqual(rel[3], "Steve Jobs co-founded Apple Inc.", "Description should be extracted correctly")


if __name__ == '__main__':
    unittest.main(verbosity=2) 