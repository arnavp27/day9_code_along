from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import re
import os
from datetime import datetime

# --- LLM Configuration ---
def get_llm(temperature: float = 0.7):
    """Get the configured LLM (Ollama or OpenAI)."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"🤖 Using Ollama: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            format="json" if temperature == 0 else None  # Request JSON format for evaluation
        )
    else:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"🤖 Using OpenAI: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature
        )


# --- Define the workflow state schema ---
class QAState(TypedDict, total=False):
    question: str
    draft_answer: Optional[str]
    evaluation: Optional[dict]
    final_answer: Optional[str]
    reflections: list[str]
    search_results: Optional[str]
    iteration_count: int
    max_iterations: int


# --- Define graph nodes ---
def generate_answer_node(state: QAState) -> QAState:
    """Generate an initial answer to the question."""
    print("\n🔵 NODE: generate_answer")
    
    llm = get_llm(temperature=0.7)
    question = state["question"]
    
    # Include search results if available
    context = ""
    if state.get("search_results"):
        context = f"\n\nSearch Results:\n{state['search_results']}\n\n"
    
    # Include reflections from previous iterations
    reflection_context = ""
    if state.get("reflections"):
        reflection_context = "\n\nPrevious Reflections:\n" + "\n".join(
            f"- {r}" for r in state["reflections"]
        )
    
    prompt = f"""Provide a detailed, accurate answer to this question:{context}{reflection_context}
Question: {question}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    draft = response.content
    
    print(f"✅ Generated draft ({len(draft)} chars)")
    
    return {
        "draft_answer": draft,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def search_tool_node(state: QAState) -> QAState:
    """Use Tavily search tool to gather additional information."""
    print("\n🔍 NODE: search_tool")
    
    # Initialize Tavily search
    search = TavilySearchResults(
        max_results=3,
        search_depth="advanced"
    )
    
    question = state["question"]
    print(f"🔎 Searching for: {question}")
    
    try:
        results = search.invoke(question)
        
        # Format search results
        formatted_results = []
        for idx, result in enumerate(results, 1):
            formatted_results.append(
                f"{idx}. {result.get('content', '')}\nSource: {result.get('url', 'N/A')}"
            )
        
        search_output = "\n\n".join(formatted_results)
        print(f"✅ Found {len(results)} search results")
        
        return {"search_results": search_output}
    
    except Exception as e:
        print(f"⚠️ Search failed: {e}")
        return {"search_results": "Search unavailable"}


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text that might contain markdown or other formatting."""
    # Try to find JSON between ```json and ``` markers
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to find any JSON object in the text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Try parsing the whole text
    try:
        return json.loads(text)
    except:
        pass
    
    return None


def evaluate_answer_node(state: QAState) -> QAState:
    """Evaluate the draft answer and return structured JSON feedback."""
    print("\n📊 NODE: evaluate_answer")
    
    llm = get_llm(temperature=0)
    
    # Simpler, more explicit prompt for Ollama
    evaluation_prompt = f"""You must respond with ONLY a JSON object, no other text.

Question: {state['question']}

Answer to evaluate: {state['draft_answer']}

Respond with this exact JSON structure:
{{
    "score": 8,
    "is_acceptable": true,
    "strengths": ["clear explanation", "good examples"],
    "weaknesses": ["could add more detail"],
    "suggestions": ["add code examples"],
    "needs_search": false
}}

Rules:
- score: number from 1-10
- is_acceptable: true if score >= 7, false otherwise
- strengths: list of 1-3 good points
- weaknesses: list of 1-3 weak points (empty list if none)
- suggestions: list of 1-3 improvements (empty list if none)
- needs_search: true only if answer lacks current information

Return ONLY the JSON object, nothing else:"""

    response = llm.invoke([HumanMessage(content=evaluation_prompt)])
    
    # Try multiple methods to extract JSON
    evaluation = extract_json_from_text(response.content)
    
    if evaluation and all(k in evaluation for k in ["score", "is_acceptable"]):
        print(f"✅ Evaluation score: {evaluation['score']}/10")
        print(f"   Acceptable: {evaluation['is_acceptable']}")
        return {"evaluation": evaluation}
    else:
        print(f"⚠️ Could not parse JSON, using fallback evaluation")
        print(f"   Raw response preview: {response.content[:200]}")
        
        # Create a reasonable fallback based on draft length and iteration
        iteration = state.get("iteration_count", 1)
        draft_len = len(state.get("draft_answer", ""))
        
        # Give it a reasonable score based on content length
        fallback_score = min(10, max(5, draft_len // 200))
        
        return {
            "evaluation": {
                "score": fallback_score,
                "is_acceptable": iteration >= 2 or fallback_score >= 7,
                "strengths": ["Answer provided"],
                "weaknesses": [] if fallback_score >= 7 else ["Could be more detailed"],
                "suggestions": [] if fallback_score >= 7 else ["Add more examples"],
                "needs_search": False
            }
        }


def reflection_node(state: QAState) -> QAState:
    """Add reflections based on evaluation feedback."""
    print("\n💭 NODE: reflection")
    
    evaluation = state.get("evaluation", {})
    reflections = state.get("reflections", [])
    
    # Create new reflection from evaluation
    new_reflection = f"Iteration {state.get('iteration_count', 0)}: "
    
    if evaluation.get("weaknesses"):
        new_reflection += f"Address: {', '.join(evaluation['weaknesses'])}. "
    
    if evaluation.get("suggestions"):
        new_reflection += f"Improve: {', '.join(evaluation['suggestions'])}"
    
    if not evaluation.get("weaknesses") and not evaluation.get("suggestions"):
        new_reflection += "Continue improving clarity and completeness"
    
    reflections.append(new_reflection)
    
    print(f"✅ Added reflection: {new_reflection}")
    
    return {"reflections": reflections}


def finalize_answer_node(state: QAState) -> QAState:
    """Finalize the answer with polish and formatting."""
    print("\n✨ NODE: finalize_answer")
    
    llm = get_llm(temperature=0.3)
    
    prompt = f"""Polish this answer for final presentation. Make it clear, concise, and well-structured.

Question: {state['question']}

Draft Answer: {state['draft_answer']}

Provide the final polished answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    final = response.content
    
    print(f"✅ Finalized answer ({len(final)} chars)")
    
    return {"final_answer": final}


# --- Conditional edge logic ---
def should_continue(state: QAState) -> Literal["search", "reflect", "finalize"]:
    """Decide next step based on evaluation."""
    evaluation = state.get("evaluation", {})
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    print(f"\n🔀 ROUTER: Iteration {iteration_count}/{max_iterations}")
    
    # Check max iterations to prevent infinite loop
    if iteration_count >= max_iterations:
        print("   ➡️ Max iterations reached → FINALIZE")
        return "finalize"
    
    # Check if answer is acceptable
    if evaluation.get("is_acceptable", False) and evaluation.get("score", 0) >= 7:
        print("   ➡️ Answer acceptable → FINALIZE")
        return "finalize"
    
    # Check if search is needed
    if evaluation.get("needs_search", False) and not state.get("search_results"):
        print("   ➡️ Search needed → SEARCH")
        return "search"
    
    # Otherwise, reflect and retry
    print("   ➡️ Needs improvement → REFLECT")
    return "reflect"


# --- Build LangGraph workflow ---
def build_workflow():
    graph = StateGraph(QAState)
    
    # Add nodes
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("search", search_tool_node)
    graph.add_node("evaluate", evaluate_answer_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("finalize", finalize_answer_node)
    
    # Add edges
    graph.add_edge(START, "generate_answer")
    graph.add_edge("generate_answer", "evaluate")
    
    # Conditional routing from evaluate
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "search": "search",
            "reflect": "reflect",
            "finalize": "finalize"
        }
    )
    
    # After search, regenerate answer
    graph.add_edge("search", "generate_answer")
    
    # After reflection, regenerate answer
    graph.add_edge("reflect", "generate_answer")
    
    # Finalize leads to end
    graph.add_edge("finalize", END)
    
    return graph.compile()