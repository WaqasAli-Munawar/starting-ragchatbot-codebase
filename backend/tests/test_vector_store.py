"""
Tests for VectorStore to identify potential issues with search functionality.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStoreSearch(unittest.TestCase):
    """Test suite for VectorStore search functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock ChromaDB dependencies
        with patch('vector_store.chromadb') as mock_chromadb, \
             patch('vector_store.SentenceTransformer'):
            
            # Mock ChromaDB client and collections
            self.mock_client = Mock()
            self.mock_course_catalog = Mock()
            self.mock_course_content = Mock()
            
            mock_chromadb.PersistentClient.return_value = self.mock_client
            self.mock_client.get_or_create_collection.side_effect = [
                self.mock_course_catalog,  # First call for catalog
                self.mock_course_content   # Second call for content
            ]
            
            # Create vector store
            self.vector_store = VectorStore("./test_chroma", "test-model", max_results=5)
    
    def test_search_successful_query(self):
        """Test successful search query"""
        # Mock ChromaDB query results
        mock_chroma_results = {
            'documents': [['Content about Python', 'Advanced Python topics']],
            'metadatas': [[
                {'course_title': 'Python Course', 'lesson_number': 1, 'chunk_index': 0},
                {'course_title': 'Python Course', 'lesson_number': 2, 'chunk_index': 1}
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        self.mock_course_content.query.return_value = mock_chroma_results
        
        # Execute search
        results = self.vector_store.search("Python basics")
        
        # Verify search was called correctly
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["Python basics"],
            n_results=5,
            where=None
        )
        
        # Verify results
        self.assertFalse(results.is_empty())
        self.assertIsNone(results.error)
        self.assertEqual(len(results.documents), 2)
        self.assertEqual(results.documents[0], 'Content about Python')
        self.assertEqual(results.metadata[0]['course_title'], 'Python Course')
    
    def test_search_with_course_filter(self):
        """Test search with course name filtering"""
        # Mock course name resolution
        mock_course_resolution = {
            'documents': [['Python Course']],
            'metadatas': [[{'title': 'Python Course'}]]
        }
        
        self.mock_course_catalog.query.return_value = mock_course_resolution
        
        # Mock content search results
        mock_content_results = {
            'documents': [['Filtered content']],
            'metadatas': [[{'course_title': 'Python Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        self.mock_course_content.query.return_value = mock_content_results
        
        # Execute search with course filter
        results = self.vector_store.search("basics", course_name="Python")
        
        # Verify course resolution was attempted
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
        
        # Verify content search used resolved course name
        content_call_args = self.mock_course_content.query.call_args
        expected_filter = {"course_title": "Python Course"}
        self.assertEqual(content_call_args[1]['where'], expected_filter)
    
    def test_search_course_not_found(self):
        """Test search when course name cannot be resolved"""
        # Mock empty course resolution
        mock_course_resolution = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        self.mock_course_catalog.query.return_value = mock_course_resolution
        
        # Execute search with non-existent course
        results = self.vector_store.search("basics", course_name="NonexistentCourse")
        
        # Should return error
        self.assertIsNotNone(results.error)
        self.assertIn("No course found matching 'NonexistentCourse'", results.error)
        self.assertTrue(results.is_empty())
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filtering"""
        mock_content_results = {
            'documents': [['Lesson 3 content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        
        self.mock_course_content.query.return_value = mock_content_results
        
        # Execute search with lesson filter
        results = self.vector_store.search("topic", lesson_number=3)
        
        # Verify filter was applied
        content_call_args = self.mock_course_content.query.call_args
        expected_filter = {"lesson_number": 3}
        self.assertEqual(content_call_args[1]['where'], expected_filter)
    
    def test_search_with_combined_filters(self):
        """Test search with both course and lesson filters"""
        # Mock course resolution
        mock_course_resolution = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        self.mock_course_catalog.query.return_value = mock_course_resolution
        
        # Mock content search
        mock_content_results = {
            'documents': [['Specific content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
            'distances': [[0.1]]
        }
        
        self.mock_course_content.query.return_value = mock_content_results
        
        # Execute search with both filters
        results = self.vector_store.search("topic", course_name="Test", lesson_number=2)
        
        # Verify combined filter
        content_call_args = self.mock_course_content.query.call_args
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        self.assertEqual(content_call_args[1]['where'], expected_filter)
    
    def test_search_chromadb_exception(self):
        """Test handling of ChromaDB exceptions"""
        # Mock ChromaDB to raise exception
        self.mock_course_content.query.side_effect = Exception("Database connection failed")
        
        # Execute search
        results = self.vector_store.search("test query")
        
        # Should return error result
        self.assertIsNotNone(results.error)
        self.assertIn("Search error:", results.error)
        self.assertIn("Database connection failed", results.error)
        self.assertTrue(results.is_empty())
    
    def test_search_empty_results(self):
        """Test handling of empty search results"""
        # Mock empty ChromaDB results
        mock_empty_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        self.mock_course_content.query.return_value = mock_empty_results
        
        # Execute search
        results = self.vector_store.search("nonexistent topic")
        
        # Should return empty results without error
        self.assertIsNone(results.error)
        self.assertTrue(results.is_empty())
        self.assertEqual(len(results.documents), 0)
    
    def test_course_name_resolution_failure(self):
        """Test handling of course name resolution exceptions"""
        # Mock course catalog to raise exception
        self.mock_course_catalog.query.side_effect = Exception("Catalog error")
        
        # Execute search with course filter
        results = self.vector_store.search("query", course_name="Test Course")
        
        # Should return error about course not found (resolution failure)
        self.assertIsNotNone(results.error)
        self.assertIn("No course found matching 'Test Course'", results.error)
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        # Create test chunks
        chunks = [
            CourseChunk("Test Course", 1, 0, "First chunk content"),
            CourseChunk("Test Course", 1, 1, "Second chunk content")
        ]
        
        # Add chunks
        self.vector_store.add_course_content(chunks)
        
        # Verify ChromaDB add was called correctly
        self.mock_course_content.add.assert_called_once()
        call_args = self.mock_course_content.add.call_args
        
        # Verify documents
        self.assertEqual(call_args[1]['documents'], ["First chunk content", "Second chunk content"])
        
        # Verify metadata
        expected_metadata = [
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1}
        ]
        self.assertEqual(call_args[1]['metadatas'], expected_metadata)
        
        # Verify IDs
        expected_ids = ["Test_Course_0", "Test_Course_1"]
        self.assertEqual(call_args[1]['ids'], expected_ids)
    
    def test_add_course_metadata(self):
        """Test adding course metadata"""
        # Create test course
        lessons = [
            Lesson(1, "Introduction", "https://example.com/lesson1"),
            Lesson(2, "Advanced Topics", None)
        ]
        course = Course("Test Course", "Test Instructor", lessons, "https://example.com/course")
        
        # Add course metadata
        self.vector_store.add_course_metadata(course)
        
        # Verify catalog add was called
        self.mock_course_catalog.add.assert_called_once()
        call_args = self.mock_course_catalog.add.call_args
        
        # Verify document
        self.assertEqual(call_args[1]['documents'], ["Test Course"])
        
        # Verify metadata structure
        metadata = call_args[1]['metadatas'][0]
        self.assertEqual(metadata['title'], "Test Course")
        self.assertEqual(metadata['instructor'], "Test Instructor")
        self.assertEqual(metadata['course_link'], "https://example.com/course")
        self.assertEqual(metadata['lesson_count'], 2)
        self.assertIn('lessons_json', metadata)
        
        # Verify ID
        self.assertEqual(call_args[1]['ids'], ["Test Course"])


class TestSearchResults(unittest.TestCase):
    """Test SearchResults helper class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'meta1': 'value1'}, {'meta2': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        self.assertEqual(results.documents, ['doc1', 'doc2'])
        self.assertEqual(results.metadata, [{'meta1': 'value1'}, {'meta2': 'value2'}])
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)
        self.assertFalse(results.is_empty())
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertTrue(results.is_empty())
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "Test error"
        results = SearchResults.empty(error_msg)
        
        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertEqual(results.error, error_msg)
        self.assertTrue(results.is_empty())


if __name__ == '__main__':
    unittest.main()