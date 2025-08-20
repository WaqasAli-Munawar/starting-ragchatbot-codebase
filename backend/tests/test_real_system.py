"""
Real system integration test to diagnose why content queries return 'query failed'.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_vector_store_data():
    """Test if vector store has course data loaded"""
    print("=== Testing Vector Store Data ===")
    
    try:
        from vector_store import VectorStore
        from config import Config
        
        # Initialize vector store with real config
        config = Config()
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check course count
        course_count = vector_store.get_course_count()
        print(f"📊 Courses in database: {course_count}")
        
        if course_count == 0:
            print("❌ NO COURSES FOUND - This is likely the main issue!")
            return False
        
        # Get course titles
        course_titles = vector_store.get_existing_course_titles()
        print(f"📚 Course titles: {course_titles}")
        
        # Test a simple search
        print("\n🔍 Testing direct vector store search...")
        results = vector_store.search("Python")
        print(f"Search results: error='{results.error}', empty={results.is_empty()}, docs={len(results.documents)}")
        
        if results.error:
            print(f"❌ Vector store search error: {results.error}")
            return False
        elif results.is_empty():
            print("⚠️  Vector store search returned no results")
            return False
        else:
            print(f"✅ Vector store search successful: {len(results.documents)} results")
            return True
            
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_tool():
    """Test CourseSearchTool directly"""
    print("\n=== Testing CourseSearchTool ===")
    
    try:
        from search_tools import CourseSearchTool
        from vector_store import VectorStore
        from config import Config
        
        config = Config()
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        
        # Test search tool
        print("🔍 Testing CourseSearchTool.execute()...")
        result = search_tool.execute("Python basics")
        
        print(f"Search tool result length: {len(result)}")
        print(f"First 200 chars: {result[:200]}")
        
        if "error" in result.lower() or "failed" in result.lower():
            print(f"❌ Search tool returned error: {result}")
            return False
        elif "no relevant content found" in result.lower():
            print("⚠️  Search tool found no relevant content")
            return False
        else:
            print("✅ Search tool working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Search tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_key():
    """Test if API key is configured"""
    print("\n=== Testing API Configuration ===")
    
    try:
        from config import Config
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
            print("❌ ANTHROPIC_API_KEY not configured properly")
            print("   Please create .env file with: ANTHROPIC_API_KEY=your-actual-key")
            return False
        
        # Don't print the actual key for security
        key_preview = config.ANTHROPIC_API_KEY[:10] + "..." if len(config.ANTHROPIC_API_KEY) > 10 else "too_short"
        print(f"✅ API key configured: {key_preview}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_tool_manager():
    """Test tool manager and registration"""
    print("\n=== Testing Tool Manager ===")
    
    try:
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
        from vector_store import VectorStore
        from config import Config
        
        config = Config()
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Create tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        outline_tool = CourseOutlineTool(vector_store)
        
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)
        
        # Test tool definitions
        tool_defs = tool_manager.get_tool_definitions()
        print(f"📋 Registered tools: {len(tool_defs)}")
        for tool_def in tool_defs:
            print(f"  - {tool_def['name']}: {tool_def['description'][:50]}...")
        
        # Test tool execution
        print("\n🔧 Testing tool execution...")
        result = tool_manager.execute_tool("search_course_content", query="Python")
        print(f"Tool execution result length: {len(result)}")
        
        if "error" in result.lower():
            print(f"❌ Tool execution failed: {result[:200]}")
            return False
        else:
            print("✅ Tool manager working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Tool manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_rag_system():
    """Test the full RAG system"""
    print("\n=== Testing Full RAG System ===")
    
    try:
        from rag_system import RAGSystem
        from config import Config
        
        config = Config()
        
        # Check if API key is set
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
            print("⚠️  Skipping RAG system test - API key not configured")
            return None
        
        # Initialize RAG system
        print("🏗️  Initializing RAG system...")
        rag_system = RAGSystem(config)
        
        # Test analytics first (doesn't need API)
        analytics = rag_system.get_course_analytics()
        print(f"📊 Course analytics: {analytics}")
        
        if analytics['total_courses'] == 0:
            print("❌ No courses loaded in RAG system")
            return False
        
        # Test query (requires API key)
        print("🤖 Testing RAG query...")
        response, sources = rag_system.query("What is Python?")
        
        print(f"Response length: {len(response)}")
        print(f"Sources: {len(sources)}")
        print(f"Response preview: {response[:200]}")
        
        if "query failed" in response.lower():
            print(f"❌ RAG system returned 'query failed': {response}")
            return False
        else:
            print("✅ RAG system working correctly")
            return True
            
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("RAG System Diagnostic Tests")
    print("=" * 50)
    
    tests = [
        ("Vector Store Data", test_vector_store_data),
        ("Search Tool", test_search_tool), 
        ("API Key", test_api_key),
        ("Tool Manager", test_tool_manager),
        ("Full RAG System", test_full_rag_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results[test_name] = result
        
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        
        print(f"\n{status} {test_name}")
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    failed_tests = [name for name, result in results.items() if result is False]
    
    if not failed_tests:
        print("✅ All tests passed! If you're still getting 'query failed', check:")
        print("   - Network connectivity to Anthropic API")
        print("   - API rate limits or quota")
        print("   - Client-side JavaScript errors")
    else:
        print("❌ Failed tests that need attention:")
        for test in failed_tests:
            print(f"   - {test}")
        
        if "Vector Store Data" in failed_tests:
            print("\n🔧 To fix Vector Store Data:")
            print("   1. Check if documents exist in /docs folder")
            print("   2. Run document ingestion: python -c \"from rag_system import RAGSystem; from config import Config; RAGSystem(Config()).add_course_folder('./docs')\"")
            print("   3. Verify ChromaDB permissions and disk space")
        
        if "API Key" in failed_tests:
            print("\n🔧 To fix API Key:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your Anthropic API key to .env")
            print("   3. Restart the application")

if __name__ == '__main__':
    main()