"""
Tests for CourseSearchTool to identify why content queries are failing.
"""
import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test suite for CourseSearchTool functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock vector store
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_with_successful_results(self):
        """Test execute method with successful search results"""
        # Mock search results with sample data
        mock_results = SearchResults(
            documents=["Sample course content about Python basics", "Advanced Python concepts"],
            metadata=[
                {"course_title": "Python Course", "lesson_number": 1},
                {"course_title": "Python Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        
        # Configure mock to return our test results
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Execute the search
        result = self.search_tool.execute("Python basics")
        
        # Verify the search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        self.assertIn("[Python Course - Lesson 1]", result)
        self.assertIn("Sample course content about Python basics", result)
        self.assertIn("[Python Course - Lesson 2]", result)
        self.assertIn("Advanced Python concepts", result)
        
        # Verify sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertIn("Python Course - Lesson 1", self.search_tool.last_sources[0])
        self.assertIn("Python Course - Lesson 2", self.search_tool.last_sources[1])
    
    def test_execute_with_error_results(self):
        """Test execute method with error from vector store"""
        # Mock search results with error
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute the search
        result = self.search_tool.execute("Python basics")
        
        # Should return the error message
        self.assertEqual(result, "Database connection failed")
    
    def test_execute_with_empty_results(self):
        """Test execute method with no search results"""
        # Mock empty search results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute the search
        result = self.search_tool.execute("Nonexistent topic")
        
        # Should return no results message
        self.assertEqual(result, "No relevant content found.")
    
    def test_execute_with_course_filter(self):
        """Test execute method with course name filtering"""
        mock_results = SearchResults(
            documents=["MCP specific content"],
            metadata=[{"course_title": "MCP Tutorial", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Execute with course filter
        result = self.search_tool.execute("MCP basics", course_name="MCP Tutorial")
        
        # Verify search was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="MCP basics",
            course_name="MCP Tutorial",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filtering"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Python Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Execute with lesson filter
        result = self.search_tool.execute("specific topic", lesson_number=3)
        
        # Verify search was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="specific topic",
            course_name=None,
            lesson_number=3
        )
    
    def test_execute_with_vector_store_exception(self):
        """Test execute method when vector store throws exception"""
        # Mock vector store to raise exception
        self.mock_vector_store.search.side_effect = Exception("Vector store crashed")
        
        # Execute the search - should not crash
        result = self.search_tool.execute("test query")
        
        # Should return error in SearchResults format
        # This tests if the vector store properly handles exceptions
        self.assertIsInstance(result, str)
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        # Verify required fields
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        
        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        # Verify required query parameter
        self.assertIn("query", schema["required"])
        self.assertIn("query", schema["properties"])
        
    def test_source_tracking_with_lesson_links(self):
        """Test that lesson links are properly tracked in sources"""
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        # Mock lesson link retrieval
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        result = self.search_tool.execute("test query")
        
        # Verify lesson link was requested
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)
        
        # Verify source includes JSON with URL
        self.assertEqual(len(self.search_tool.last_sources), 1)
        source = self.search_tool.last_sources[0]
        self.assertIn("https://example.com/lesson1", source)


if __name__ == '__main__':
    unittest.main()