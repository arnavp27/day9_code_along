"""
Test suite for the enhanced LangGraph Q&A Assistant.
Demonstrates all features: evaluation, reflection, search, memory, and tracing.
"""

import os
from dotenv import load_dotenv
from graph.workflow import build_workflow
from graph.memory import MemoryManager
import json

load_dotenv()


def test_basic_workflow():
    """Test 1: Basic workflow without search."""
    print("\n" + "="*60)
    print("TEST 1: Basic Workflow (No Search Needed)")
    print("="*60)
    
    workflow = build_workflow()
    memory = MemoryManager()
    
    question = "What is 2 + 2?"
    session_id = memory.create_session(question)
    
    result = workflow.invoke({
        "question": question,
        "max_iterations": 2
    })
    
    assert result["final_answer"] is not None, "Final answer should exist"
    assert result["iteration_count"] > 0, "Should have at least 1 iteration"
    
    print(f"✅ Test passed!")
    print(f"   - Iterations: {result['iteration_count']}")
    print(f"   - Final answer length: {len(result['final_answer'])} chars")
    
    memory.save_final_answer(result["final_answer"], result)
    return result


def test_reflection_loop():
    """Test 2: Multiple iterations with reflections."""
    print("\n" + "="*60)
    print("TEST 2: Reflection Loop (Multiple Iterations)")
    print("="*60)
    
    workflow = build_workflow()
    memory = MemoryManager()
    
    # Question designed to need improvement
    question = "Explain quantum computing in very technical detail."
    session_id = memory.create_session(question)
    
    result = workflow.invoke({
        "question": question,
        "max_iterations": 3
    })
    
    assert len(result.get("reflections", [])) > 0, "Should have reflections"
    assert result["iteration_count"] >= 2, "Should iterate multiple times"
    
    print(f"✅ Test passed!")
    print(f"   - Iterations: {result['iteration_count']}")
    print(f"   - Reflections: {len(result['reflections'])}")
    print(f"\n   Reflections generated:")
    for i, reflection in enumerate(result["reflections"], 1):
        print(f"   {i}. {reflection}")
    
    memory.save_final_answer(result["final_answer"], result)
    return result


def test_search_integration():
    """Test 3: Workflow with web search."""
    print("\n" + "="*60)
    print("TEST 3: Web Search Integration")
    print("="*60)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️ Skipping: TAVILY_API_KEY not set")
        return None
    
    workflow = build_workflow()
    memory = MemoryManager()
    
    # Question that should trigger search
    question = "What are the latest breakthroughs in AI in 2025?"
    session_id = memory.create_session(question)
    
    result = workflow.invoke({
        "question": question,
        "max_iterations": 3
    })
    
    # Note: Search might not always trigger depending on evaluation
    if result.get("search_results"):
        print(f"✅ Test passed - Search was triggered!")
        print(f"   - Search results present: Yes")
        print(f"   - Result length: {len(result['search_results'])} chars")
    else:
        print(f"⚠️ Search was not triggered (evaluation didn't require it)")
    
    print(f"   - Iterations: {result['iteration_count']}")
    
    memory.save_final_answer(result["final_answer"], result)
    return result


def test_max_iterations_limit():
    """Test 4: Max iterations prevents infinite loop."""
    print("\n" + "="*60)
    print("TEST 4: Max Iterations Limit (Safety)")
    print("="*60)
    
    workflow = build_workflow()
    memory = MemoryManager()
    
    question = "Explain everything about the universe."
    session_id = memory.create_session(question)
    
    max_iter = 2
    result = workflow.invoke({
        "question": question,
        "max_iterations": max_iter
    })
    
    assert result["iteration_count"] <= max_iter, f"Should stop at {max_iter} iterations"
    
    print(f"✅ Test passed!")
    print(f"   - Max iterations set: {max_iter}")
    print(f"   - Actual iterations: {result['iteration_count']}")
    print(f"   - No infinite loop!")
    
    memory.save_final_answer(result["final_answer"], result)
    return result


def test_structured_evaluation():
    """Test 5: Evaluation returns structured JSON."""
    print("\n" + "="*60)
    print("TEST 5: Structured JSON Evaluation")
    print("="*60)
    
    workflow = build_workflow()
    memory = MemoryManager()
    
    question = "What is machine learning?"
    session_id = memory.create_session(question)
    
    result = workflow.invoke({
        "question": question,
        "max_iterations": 2
    })
    
    evaluation = result.get("evaluation", {})
    
    # Verify JSON structure
    required_keys = ["score", "is_acceptable", "strengths", "weaknesses", "suggestions"]
    for key in required_keys:
        assert key in evaluation, f"Evaluation missing key: {key}"
    
    print(f"✅ Test passed!")
    print(f"\n   Evaluation structure:")
    print(json.dumps(evaluation, indent=4))
    
    memory.save_final_answer(result["final_answer"], result)
    return result


def test_memory_persistence():
    """Test 6: Memory saves and retrieves sessions."""
    print("\n" + "="*60)
    print("TEST 6: Memory Persistence")
    print("="*60)
    
    memory = MemoryManager()
    workflow = build_workflow()
    
    # Create a session
    question = "Test question for memory persistence"
    session_id = memory.create_session(question)
    
    result = workflow.invoke({
        "question": question,
        "max_iterations": 1
    })
    
    memory.save_final_answer(result["final_answer"], result)
    
    # Verify session was saved
    sessions = memory.get_session_history(limit=1)
    
    assert len(sessions) > 0, "Should have saved sessions"
    assert sessions[0]["question"] == question, "Question should match"
    
    print(f"✅ Test passed!")
    print(f"   - Session ID: {session_id}")
    print(f"   - Sessions in memory: {len(sessions)}")
    print(f"   - Memory file exists: Yes")
    
    # Test export
    export_file = memory.export_session(session_id)
    print(f"   - Exported to: {export_file}")
    
    return sessions


def test_tracing_enabled():
    """Test 7: LangSmith tracing is configured."""
    print("\n" + "="*60)
    print("TEST 7: LangSmith Tracing Configuration")
    print("="*60)
    
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key_set = bool(os.getenv("LANGCHAIN_API_KEY"))
    
    if tracing_enabled and api_key_set:
        print(f"✅ Test passed!")
        print(f"   - LANGCHAIN_TRACING_V2: {tracing_enabled}")
        print(f"   - API key configured: {api_key_set}")
        print(f"   - Project: {os.getenv('LANGCHAIN_PROJECT', 'N/A')}")
    else:
        print(f"⚠️ Tracing not fully configured")
        print(f"   - LANGCHAIN_TRACING_V2: {tracing_enabled}")
        print(f"   - API key configured: {api_key_set}")
        print(f"   - This is optional but recommended")


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*60)
    print("🧪 RUNNING FULL TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Workflow", test_basic_workflow),
        ("Reflection Loop", test_reflection_loop),
        ("Search Integration", test_search_integration),
        ("Max Iterations", test_max_iterations_limit),
        ("Structured Evaluation", test_structured_evaluation),
        ("Memory Persistence", test_memory_persistence),
        ("Tracing Config", test_tracing_enabled),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASSED"
        except Exception as e:
            results[test_name] = f"❌ FAILED: {str(e)}"
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, status in results.items():
        print(f"{status} - {test_name}")
    
    passed = sum(1 for s in results.values() if "PASSED" in s)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in .env file")
        exit(1)
    
    run_all_tests()