import os
from dotenv import load_dotenv
from graph.workflow import build_workflow
from graph.memory import MemoryManager

# Load environment variables
load_dotenv()

# Disable LangSmith tracing by default (enable only if explicitly set)
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    print("✅ LangSmith tracing enabled")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def run_qa_workflow(question: str, max_iterations: int = 3):
    """Execute the QA workflow with memory persistence."""
    
    # Initialize memory manager
    memory = MemoryManager()
    session_id = memory.create_session(question)
    
    print("\n" + "="*60)
    print("🚀 STARTING QA WORKFLOW")
    print("="*60)
    print(f"Question: {question}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Session ID: {session_id}")
    print("="*60 + "\n")
    
    # Build workflow
    workflow = build_workflow()
    
    # Initial state
    initial_state = {
        "question": question,
        "max_iterations": max_iterations,
        "iteration_count": 0,
        "reflections": []
    }
    
    # Execute workflow
    try:
        # Stream the workflow execution to see each step
        result = None
        for step_output in workflow.stream(initial_state):
            # Log each state transition
            for node_name, node_state in step_output.items():
                print(f"\n📍 Executed: {node_name}")
                memory.log_state(node_name, node_state)
                result = node_state
        
        # Save final answer
        if result and result.get("final_answer"):
            memory.save_final_answer(result["final_answer"], result)
        
        print("\n" + "="*60)
        print("✅ WORKFLOW COMPLETED")
        print("="*60)
        
        # Print session summary
        memory.print_session_summary()
        
        # Export session
        memory.export_session()
        
        return result
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point with interactive options."""
    print("\n" + "="*60)
    print("🤖 LangGraph Q&A Assistant with Ollama")
    print("="*60)
    
    # Check for LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        print(f"✅ Using Ollama with model: {model_name}")
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print(f"❌ Missing OPENAI_API_KEY for OpenAI provider")
            return
        print(f"✅ Using OpenAI")
    
    # Check optional features
    if not os.getenv("TAVILY_API_KEY"):
        print(f"ℹ️  Web search disabled (no TAVILY_API_KEY)")
    else:
        print(f"✅ Web search enabled")
    
    # Interactive mode
    while True:
        print("\n" + "="*60)
        print("Options:")
        print("1. Ask a new question")
        print("2. View recent sessions")
        print("3. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            question = input("\n📝 Ask me a question: ").strip()
            if not question:
                question = "What are the benefits of renewable energy?"
                print(f"Using default question: {question}")
            
            # Ask for max iterations
            max_iter_input = input("Max iterations (default 3): ").strip()
            max_iterations = int(max_iter_input) if max_iter_input else 3
            
            # Run workflow
            result = run_qa_workflow(question, max_iterations)
            
            if result:
                print("\n" + "="*60)
                print("📄 FINAL ANSWER")
                print("="*60)
                print(result.get("final_answer", "No answer generated"))
                print("="*60)
                
                # Show metadata
                print("\n📊 Workflow Statistics:")
                print(f"   - Total Iterations: {result.get('iteration_count', 0)}")
                print(f"   - Reflections Generated: {len(result.get('reflections', []))}")
                print(f"   - Search Used: {'Yes' if result.get('search_results') else 'No'}")
                if result.get('evaluation'):
                    print(f"   - Final Score: {result['evaluation'].get('score', 'N/A')}/10")
        
        elif choice == "2":
            memory = MemoryManager()
            sessions = memory.get_session_history(limit=5)
            
            if not sessions:
                print("\n❌ No sessions found")
            else:
                print("\n📚 Recent Sessions:")
                print("="*60)
                for i, session in enumerate(sessions, 1):
                    print(f"{i}. [{session['session_id']}]")
                    print(f"   Q: {session['question'][:60]}...")
                    print(f"   Iterations: {session['metadata'].get('total_iterations', 0)}")
                    print()
        
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()