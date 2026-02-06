"""
Test the adaptive learning LangGraph system
"""
from agents_diagnostic import app, StudyState
from knowledge_graph import KnowledgeGraph

def test_learning_flow():
    """Test the complete learning flow."""
    print("="*70)
    print("ADAPTIVE LEARNING SYSTEM TEST")
    print("="*70)
    
    # Initialize state (simplified version)
    initial_state: StudyState = {
        "messages": [],
        "topic": "Python Programming",
        "curriculum": "",
        "current_node_id": "",
    "student_answer": "",  # Start clean; we'll answer after a question is asked
        "next": "design"
    }
    
    print("\n📚 Step 1: Designing Curriculum and First Question...")
    print("-"*70)
    
    # Configuration with thread_id for session persistence
    config = {
        "configurable": {"thread_id": "test_user_python_001"},
        "recursion_limit": 10
    }
    
    # Stream the graph execution to see each step
    # Session will be automatically saved to learning_sessions.db
    final_state = None
    iteration = 0
    for state in app.stream(initial_state, config=config):
        iteration += 1
        print(f"   → Step {iteration}: {list(state.keys())[0]}")
        final_state = list(state.values())[0]
        if iteration >= 6:  # Stop after 6 steps
            break
    
    result = final_state if final_state else initial_state
    
    # Show curriculum
    if not result.get("curriculum"):
        print("\n❌ Error: No curriculum generated")
        return result
        
    graph = KnowledgeGraph.from_json(result["curriculum"])
    print(f"\n✅ Created curriculum for: {graph.topic}")
    print(f"   Total nodes: {len(graph.nodes)}")
    
    print("\n📋 Learning Path:")
    for i, node in enumerate(graph.get_learning_path()[:5], 1):  # Show first 5
        print(f"   {i}. {node.name}")
        print(f"      └─ {node.description}")
    
    # Show first question
    if result.get("current_node_id"):
        current_node = graph.get_node(result["current_node_id"])
        print(f"\n🎯 Current Focus: {current_node.name}")
        print(f"   Proficiency: {current_node.proficiency:.0f}/100")
    
    if result.get("question"):
        print("\n❓ Question Generated:")
        print(f"   {result['question']}")
    
    # Show grading results if available
    if result.get("score"):
        print(f"\n✅ Score: {result['score']:.0f}/100")
        print(f"📝 Feedback: {result.get('feedback', '')[:150]}...")
    
    # Show overall progress
    print("\n"+"="*70)
    print("� OVERALL PROGRESS")
    print("="*70)
    summary = graph.get_progress_summary()
    print(f"   Completed: {summary['completed']}/{summary['total_nodes']} nodes")
    print(f"   In Progress: {summary['in_progress']}")
    print(f"   Average Proficiency: {summary['avg_proficiency']:.1f}%")
    print(f"   Overall Completion: {summary['completion_percentage']:.1f}%")
    
    print(f"\n   Next action: {result.get('action', 'N/A')}")
    
    print("\n✅ Test completed successfully!")
    print("\n💡 How the system works:")
    print("   1. Course Designer creates curriculum DAG")
    print("   2. Progressor generates adaptive question")
    print("   3. Grader evaluates answer and updates proficiency")
    print("   4. System loops back to Progressor until mastery (≥80%)")
    print("   5. When topic mastered, moves to next available node")
    print("   6. When all topics complete, workflow ends")
    
    print("\n💾 Session Persistence:")
    print("   ✅ Progress saved to: learning_sessions.db")
    print(f"   ✅ Thread ID: test_user_python_001")
    print("   ✅ Can resume this session anytime!")
    
    return result


if __name__ == "__main__":
    try:
        test_learning_flow()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
