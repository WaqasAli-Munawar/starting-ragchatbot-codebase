"""
Test runner script to execute all RAG system tests and identify failing components.
"""
import unittest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
from test_course_search_tool import TestCourseSearchTool
from test_ai_generator import TestAIGenerator  
from test_rag_system import TestRAGSystemIntegration, TestRAGSystemRealIntegration
from test_vector_store import TestVectorStoreSearch, TestSearchResults


def run_component_tests():
    """Run tests for each component individually and report results"""
    
    components = [
        ("CourseSearchTool", TestCourseSearchTool),
        ("AIGenerator", TestAIGenerator),
        ("VectorStore", TestVectorStoreSearch),
        ("SearchResults", TestSearchResults),
        ("RAG System Integration", TestRAGSystemIntegration),
        ("RAG System Real Integration", TestRAGSystemRealIntegration)
    ]
    
    results = {}
    
    for component_name, test_class in components:
        print(f"\n{'='*60}")
        print(f"Testing {component_name}")
        print(f"{'='*60}")
        
        # Create test suite for this component
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Store results
        results[component_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'failure_details': result.failures,
            'error_details': result.errors
        }
        
        # Print summary for this component
        print(f"\n{component_name} Summary:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Success rate: {results[component_name]['success_rate']:.1f}%")
        
        if result.failures:
            print(f"  Failures:")
            for test, traceback in result.failures:
                print(f"    - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
        
        if result.errors:
            print(f"  Errors:")
            for test, traceback in result.errors:
                error_line = traceback.split('\n')[-2] if traceback.split('\n') else 'Unknown error'
                print(f"    - {test}: {error_line}")
    
    return results


def print_overall_summary(results):
    """Print overall test summary and recommendations"""
    print(f"\n{'='*60}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = sum(r['tests_run'] for r in results.values())
    total_failures = sum(r['failures'] for r in results.values()) 
    total_errors = sum(r['errors'] for r in results.values())
    overall_success = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Overall success rate: {overall_success:.1f}%")
    
    print(f"\nComponent Performance:")
    for component, result in results.items():
        status = "✅ PASS" if result['failures'] == 0 and result['errors'] == 0 else "❌ FAIL"
        print(f"  {status} {component}: {result['success_rate']:.1f}% ({result['tests_run']} tests)")
    
    # Identify likely problem areas
    print(f"\n{'='*60}")
    print("DIAGNOSTIC RECOMMENDATIONS")
    print(f"{'='*60}")
    
    failing_components = [name for name, result in results.items() 
                         if result['failures'] > 0 or result['errors'] > 0]
    
    if not failing_components:
        print("✅ All components pass their unit tests!")
        print("   The 'query failed' issue may be:")
        print("   - Related to real API calls (Anthropic API)")
        print("   - Missing or corrupted course data")
        print("   - Configuration issues (API keys, etc.)")
        print("   - Network/connectivity problems")
    else:
        print("❌ The following components have test failures:")
        for component in failing_components:
            print(f"   - {component}")
        
        # Specific recommendations based on which components fail
        if "VectorStore" in failing_components:
            print("\n🔍 VectorStore Issues Detected:")
            print("   - Check ChromaDB installation and permissions")
            print("   - Verify vector database files exist and are readable")
            print("   - Check if course documents were properly ingested")
            
        if "CourseSearchTool" in failing_components:
            print("\n🔍 CourseSearchTool Issues Detected:")
            print("   - Vector store search may be returning invalid results")
            print("   - Check if search results formatting is correct")
            print("   - Verify source tracking is working")
            
        if "AIGenerator" in failing_components:
            print("\n🔍 AIGenerator Issues Detected:")
            print("   - Check Anthropic API key is valid and set")
            print("   - Verify tool definitions are correctly formatted")
            print("   - Check if Claude model is responding to tool calls")
            
        if any("RAG System" in comp for comp in failing_components):
            print("\n🔍 RAG System Integration Issues Detected:")
            print("   - Check component wiring and initialization")
            print("   - Verify all dependencies are properly injected")
            print("   - Check error handling in the integration layer")


if __name__ == '__main__':
    print("RAG System Diagnostic Test Suite")
    print("This will test each component to identify why content queries return 'query failed'")
    print("")
    
    try:
        results = run_component_tests()
        print_overall_summary(results)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR running tests: {e}")
        import traceback
        traceback.print_exc()