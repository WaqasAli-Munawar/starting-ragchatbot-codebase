"""
Tests for AIGenerator to verify it correctly calls CourseSearchTool for content queries.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator(unittest.TestCase):
    """Test suite for AI Generator tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock API key and model
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"
        
        # Create AI generator with mocked Anthropic client
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a general knowledge response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response
        result = self.ai_generator.generate_response("What is Python?")
        
        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        
        # Verify basic parameters
        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(call_args["temperature"], 0)
        self.assertEqual(call_args["max_tokens"], 800)
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "What is Python?")
        
        # Verify no tools were passed
        self.assertNotIn("tools", call_args)
        
        # Verify result
        self.assertEqual(result, "This is a general knowledge response")
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Mock tool definitions
        mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        # Mock Claude response without tool use
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a direct response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response
        result = self.ai_generator.generate_response(
            "What is the capital of France?",
            tools=mock_tools
        )
        
        # Verify API was called with tools
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["tools"], mock_tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})
        
        # Verify result (no tool execution)
        self.assertEqual(result, "This is a direct response")
    
    def test_generate_response_with_tool_use(self):
        """Test response generation when Claude decides to use tools"""
        # Mock tool definitions
        mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result: Python basics explained"
        
        # Mock initial Claude response with tool use
        mock_tool_use_block = Mock()
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.name = "search_course_content"
        mock_tool_use_block.input = {"query": "Python basics"}
        mock_tool_use_block.id = "tool_use_123"
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use_block]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final Claude response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on the course materials, Python is...")]
        
        # Configure mock client to return different responses
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,  # First call triggers tool use
            mock_final_response     # Second call after tool execution
        ]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "What are Python basics?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify two API calls were made
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Verify final result
        self.assertEqual(result, "Based on the course materials, Python is...")
    
    def test_tool_execution_with_multiple_tools(self):
        """Test handling multiple tool calls in one response"""
        mock_tools = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Mock multiple tool use blocks
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.input = {"query": "query1"}
        tool_use_1.id = "tool_1"
        
        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"  
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "query2"}
        tool_use_2.id = "tool_2"
        
        mock_initial_response = Mock()
        mock_initial_response.content = [tool_use_1, tool_use_2]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Test query",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Check tool execution calls
        calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("search_course_content",))
        self.assertEqual(calls[0][1], {"query": "query1"})
        self.assertEqual(calls[1][0], ("search_course_content",))
        self.assertEqual(calls[1][1], {"query": "query2"})
    
    def test_tool_execution_failure(self):
        """Test handling when tool execution fails"""
        mock_tools = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed"
        
        # Mock tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I apologize, there was an error")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Generate response
        result = self.ai_generator.generate_response(
            "Test query",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Tool should still be called even if it fails
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Should still return a response
        self.assertEqual(result, "I apologize, there was an error")
    
    def test_conversation_history_inclusion(self):
        """Test that conversation history is properly included"""
        history = "User: Previous question\nAssistant: Previous answer"
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response with history
        result = self.ai_generator.generate_response(
            "Follow-up question",
            conversation_history=history
        )
        
        # Verify history was included in system prompt
        call_args = self.mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        self.assertIn("Previous conversation:", system_content)
        self.assertIn(history, system_content)
        
    def test_system_prompt_content(self):
        """Test that system prompt contains proper tool guidance"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response
        
        # Generate response
        self.ai_generator.generate_response("Test query")
        
        # Verify system prompt content
        call_args = self.mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        
        # Check for key guidance elements
        self.assertIn("search_course_content", system_content)
        self.assertIn("get_course_outline", system_content)
        self.assertIn("One search per query maximum", system_content)
        self.assertIn("Course-specific content questions", system_content)


if __name__ == '__main__':
    unittest.main()