"""
Tests for RAG system integration to identify why content queries are returning 'query failed'.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemIntegration(unittest.TestCase):
    """Test suite for RAG system query handling"""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_chroma"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.MAX_HISTORY = 2
        
        # Patch all dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr, \
             patch('rag_system.ToolManager') as mock_tool_mgr, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:
            
            # Set up mocks
            self.mock_vector_store_instance = Mock()
            self.mock_ai_generator_instance = Mock()
            self.mock_session_manager_instance = Mock()
            self.mock_tool_manager_instance = Mock()
            self.mock_search_tool_instance = Mock()
            self.mock_outline_tool_instance = Mock()
            
            # Configure mock constructors
            mock_vector_store.return_value = self.mock_vector_store_instance
            mock_ai_gen.return_value = self.mock_ai_generator_instance
            mock_session_mgr.return_value = self.mock_session_manager_instance
            mock_tool_mgr.return_value = self.mock_tool_manager_instance
            mock_search_tool.return_value = self.mock_search_tool_instance
            mock_outline_tool.return_value = self.mock_outline_tool_instance
            
            # Create RAG system
            self.rag_system = RAGSystem(self.mock_config)
    
    def test_query_successful_content_search(self):
        """Test successful content query handling"""
        # Mock session manager
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        
        # Mock AI generator to return successful response
        self.mock_ai_generator_instance.generate_response.return_value = "Here's information about Python basics..."
        
        # Mock tool manager to return sources
        self.mock_tool_manager_instance.get_last_sources.return_value = [
            "Python Course - Lesson 1",
            "Python Course - Lesson 2"
        ]
        self.mock_tool_manager_instance.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        
        # Execute query
        response, sources = self.rag_system.query("What are Python basics?")
        
        # Verify AI generator was called correctly
        self.mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        
        # Check the query was properly formatted
        query_arg = call_args[1]['query']
        self.assertIn("What are Python basics?", query_arg)
        
        # Check tools were provided
        self.assertIsNotNone(call_args[1]['tools'])
        self.assertIsNotNone(call_args[1]['tool_manager'])
        
        # Verify response and sources
        self.assertEqual(response, "Here's information about Python basics...")
        self.assertEqual(len(sources), 2)
        self.assertIn("Python Course - Lesson 1", sources)
        
        # Verify source management
        self.mock_tool_manager_instance.get_last_sources.assert_called_once()
        self.mock_tool_manager_instance.reset_sources.assert_called_once()
    
    def test_query_with_session_history(self):
        """Test query handling with conversation history"""
        session_id = "test_session_123"
        history = "User: Previous question\nAssistant: Previous answer"
        
        # Mock session manager
        self.mock_session_manager_instance.get_conversation_history.return_value = history
        self.mock_ai_generator_instance.generate_response.return_value = "Follow-up response"
        self.mock_tool_manager_instance.get_last_sources.return_value = []
        
        # Execute query with session
        response, sources = self.rag_system.query("Follow-up question", session_id=session_id)
        
        # Verify history was retrieved and used
        self.mock_session_manager_instance.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received history
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        self.assertEqual(call_args[1]['conversation_history'], history)
        
        # Verify session was updated
        self.mock_session_manager_instance.add_exchange.assert_called_once_with(
            session_id, "Answer this question about course materials: Follow-up question", "Follow-up response"
        )
    
    def test_query_ai_generator_failure(self):
        """Test handling when AI generator fails"""
        # Mock AI generator to raise exception
        self.mock_ai_generator_instance.generate_response.side_effect = Exception("API Error")
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        
        # Execute query - should not crash
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")
        
        # Verify the exception propagated (RAG system doesn't handle AI generator exceptions)
        self.assertIn("API Error", str(context.exception))
    
    def test_query_tool_manager_failure(self):
        """Test handling when tool manager operations fail"""
        # Mock successful AI response but failing tool operations
        self.mock_ai_generator_instance.generate_response.return_value = "Response"
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        
        # Mock tool manager failures
        self.mock_tool_manager_instance.get_tool_definitions.side_effect = Exception("Tool definition error")
        
        # Execute query - should crash because tool definitions are critical
        with self.assertRaises(Exception):
            self.rag_system.query("Test query")
    
    def test_query_sources_retrieval_failure(self):
        """Test handling when source retrieval fails"""
        # Mock successful AI response
        self.mock_ai_generator_instance.generate_response.return_value = "Response"
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        self.mock_tool_manager_instance.get_tool_definitions.return_value = []
        
        # Mock source retrieval failure
        self.mock_tool_manager_instance.get_last_sources.side_effect = Exception("Source error")
        
        # Execute query - should handle gracefully
        with self.assertRaises(Exception):
            self.rag_system.query("Test query")
    
    def test_query_prompt_formatting(self):
        """Test that query is properly formatted for AI generator"""
        test_query = "What is machine learning?"
        
        self.mock_ai_generator_instance.generate_response.return_value = "ML explanation"
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        self.mock_tool_manager_instance.get_last_sources.return_value = []
        self.mock_tool_manager_instance.get_tool_definitions.return_value = []
        
        # Execute query
        self.rag_system.query(test_query)
        
        # Verify prompt formatting
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        query_arg = call_args[1]['query']
        
        # Should contain the formatted prompt
        self.assertIn("Answer this question about course materials:", query_arg)
        self.assertIn(test_query, query_arg)
    
    def test_empty_query_handling(self):
        """Test handling of empty or None queries"""
        self.mock_ai_generator_instance.generate_response.return_value = "Please provide a question"
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        self.mock_tool_manager_instance.get_last_sources.return_value = []
        self.mock_tool_manager_instance.get_tool_definitions.return_value = []
        
        # Test empty string
        response, sources = self.rag_system.query("")
        self.assertIsInstance(response, str)
        self.assertIsInstance(sources, list)
        
        # Test None (should not crash but may raise TypeError)
        try:
            response, sources = self.rag_system.query(None)
        except TypeError:
            pass  # Expected for None input
    
    def test_tool_definitions_passed_correctly(self):
        """Test that tool definitions are correctly passed to AI generator"""
        mock_tool_defs = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        
        self.mock_tool_manager_instance.get_tool_definitions.return_value = mock_tool_defs
        self.mock_ai_generator_instance.generate_response.return_value = "Response"
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        self.mock_tool_manager_instance.get_last_sources.return_value = []
        
        # Execute query
        self.rag_system.query("Test query")
        
        # Verify tool definitions and manager were passed
        call_args = self.mock_ai_generator_instance.generate_response.call_args
        self.assertEqual(call_args[1]['tools'], mock_tool_defs)
        self.assertEqual(call_args[1]['tool_manager'], self.mock_tool_manager_instance)
    
    def test_course_analytics(self):
        """Test course analytics functionality"""
        # Mock vector store analytics
        self.mock_vector_store_instance.get_course_count.return_value = 5
        self.mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Python Course", "MCP Tutorial", "Web Development"
        ]
        
        # Get analytics
        analytics = self.rag_system.get_course_analytics()
        
        # Verify analytics structure
        self.assertIn("total_courses", analytics)
        self.assertIn("course_titles", analytics)
        self.assertEqual(analytics["total_courses"], 5)
        self.assertEqual(len(analytics["course_titles"]), 3)


class TestRAGSystemRealIntegration(unittest.TestCase):
    """Integration tests that test actual component interactions"""
    
    def test_real_tool_registration(self):
        """Test that tools are actually registered correctly in real RAG system"""
        # This test uses real objects but with minimal config to verify registration
        mock_config = Mock()
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        mock_config.CHUNK_SIZE = 800
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = "./test_chroma"
        mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        mock_config.MAX_RESULTS = 5
        mock_config.MAX_HISTORY = 2
        
        # Test tool registration by checking if we can create the system without errors
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            try:
                rag_system = RAGSystem(mock_config)
                
                # Verify tools are registered
                tool_names = list(rag_system.tool_manager.tools.keys())
                self.assertIn("search_course_content", tool_names)
                self.assertIn("get_course_outline", tool_names)
                
                # Verify tool definitions can be retrieved
                tool_defs = rag_system.tool_manager.get_tool_definitions()
                self.assertEqual(len(tool_defs), 2)
                
                # Verify each tool has required fields
                for tool_def in tool_defs:
                    self.assertIn("name", tool_def)
                    self.assertIn("description", tool_def)
                    self.assertIn("input_schema", tool_def)
                    
            except Exception as e:
                self.fail(f"RAG system initialization failed: {e}")


if __name__ == '__main__':
    unittest.main()