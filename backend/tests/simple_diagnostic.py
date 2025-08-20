"""
Simple diagnostic test without Unicode characters.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_system():
    """Test the system components one by one"""
    
    print("RAG System Diagnostic Test")
    print("=" * 40)
    
    # Test 1: Check if courses are loaded
    print("\n1. Testing Vector Store...")
    try:
        from vector_store import VectorStore
        from config import Config
        
        config = Config()
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        course_count = vector_store.get_course_count()
        print(f"   Courses in database: {course_count}")
        
        if course_count == 0:
            print("   ERROR: No courses found in database!")
            print("   This is likely the main issue causing 'query failed'")
            return False
        
        course_titles = vector_store.get_existing_course_titles()
        print(f"   Course titles: {course_titles}")
        
        # Test search
        results = vector_store.search("Python")
        if results.error:
            print(f"   ERROR: Vector search failed: {results.error}")
            return False
        elif results.is_empty():
            print("   WARNING: Search returned no results")
            return False
        else:
            print(f"   SUCCESS: Found {len(results.documents)} search results")
        
    except Exception as e:
        print(f"   ERROR: Vector store test failed: {e}")
        return False
    
    # Test 2: Check API key
    print("\n2. Testing API Key...")
    try:
        from config import Config
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
            print("   ERROR: API key not configured")
            print("   Please create .env file with ANTHROPIC_API_KEY=your-key")
            return False
        
        key_preview = config.ANTHROPIC_API_KEY[:8] + "..."
        print(f"   SUCCESS: API key configured ({key_preview})")
        
    except Exception as e:
        print(f"   ERROR: Config test failed: {e}")
        return False
    
    # Test 3: Test CourseSearchTool
    print("\n3. Testing CourseSearchTool...")
    try:
        from search_tools import CourseSearchTool
        
        search_tool = CourseSearchTool(vector_store)
        result = search_tool.execute("Python")
        
        if "error" in result.lower() or "failed" in result.lower():
            print(f"   ERROR: Search tool failed: {result[:100]}...")
            return False
        elif "no relevant content found" in result.lower():
            print("   WARNING: Search tool found no content")
            return False
        else:
            print(f"   SUCCESS: Search tool returned {len(result)} characters")
            
    except Exception as e:
        print(f"   ERROR: Search tool test failed: {e}")
        return False
    
    # Test 4: Test RAG System
    print("\n4. Testing RAG System...")
    try:
        from rag_system import RAGSystem
        
        rag_system = RAGSystem(config)
        analytics = rag_system.get_course_analytics()
        print(f"   Analytics: {analytics}")
        
        if analytics['total_courses'] == 0:
            print("   ERROR: RAG system shows no courses")
            return False
        
        print("   SUCCESS: RAG system initialized correctly")
        
    except Exception as e:
        print(f"   ERROR: RAG system test failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = test_system()
    
    print("\n" + "=" * 40)
    if success:
        print("RESULT: All core components working!")
        print("If still getting 'query failed', check:")
        print("- Network connection to Anthropic API")
        print("- API rate limits")
        print("- Browser console for JavaScript errors")
    else:
        print("RESULT: Found issues that need fixing")
        print("The diagnostic above shows what needs attention")