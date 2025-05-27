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

from const import (
    entity_pattern, relationship_pattern
)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_regex_patterns(self):
        """Test that regex patterns are valid."""
        # Test entity pattern
        try:
            re.compile(entity_pattern)
        except re.error as e:
            self.fail(f"Invalid entity_pattern regex: {e}")
        
        # Test relationship pattern
        try:
            re.compile(relationship_pattern)
        except re.error as e:
            self.fail(f"Invalid relationship_pattern regex: {e}")
    
    def test_entity_pattern_matching(self):
        """Test entity pattern matching functionality."""
        test_text = '("entity"$$$$"Apple Inc"$$$$"Company"$$$$"A technology company")'
        matches = re.findall(entity_pattern, test_text)
        
        self.assertEqual(len(matches), 1, "Should find exactly one entity match")
        entity = matches[0]
        self.assertEqual(entity[0], "Apple Inc", "Entity name should be extracted correctly")
        self.assertEqual(entity[1], "Company", "Entity type should be extracted correctly")
        self.assertEqual(entity[2], "A technology company", "Entity description should be extracted correctly")
    
    def test_relationship_pattern_matching(self):
        """Test relationship pattern matching functionality."""
        test_text = '("relationship"$$$$"Apple Inc"$$$$"iPhone"$$$$"manufactures"$$$$"Apple manufactures iPhone")'
        matches = re.findall(relationship_pattern, test_text)
        
        self.assertEqual(len(matches), 1, "Should find exactly one relationship match")
        rel = matches[0]
        self.assertEqual(rel[0], "Apple Inc", "Source entity should be extracted correctly")
        self.assertEqual(rel[1], "iPhone", "Target entity should be extracted correctly")
        self.assertEqual(rel[2], "manufactures", "Relation should be extracted correctly")
        self.assertEqual(rel[3], "Apple manufactures iPhone", "Description should be extracted correctly")


if __name__ == '__main__':
    unittest.main(verbosity=2) 