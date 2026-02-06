"""
Test resuming a persisted learning session
"""
from agents_diagnostic import app
from knowledge_graph import KnowledgeGraph

def test_resume_session():
    """Resume an existing learning session from the database."""
    print("="*70)
    print("RESUME LEARNING SESSION TEST")
    print("="*70)
    
    # The thread_id from our previous session
    thread_id = "test_user_python_001"
    
    print(f"\n🔄 Attempting to resume session: {thread_id}")
    print("-"*70)
    
    # Get the current state from the checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get the last state
        state = app.get_state(config)
        
        if not state or not state.values:
            print("\n❌ No saved session found!")
            print("   Run test_study_system.py first to create a session.")
            return
        
        current_state = state.values
        
        print("\n✅ Session found!")
        print(f"   Checkpoint ID: {state.config.get('configurable', {}).get('checkpoint_id', 'N/A')}")
        
        # Show current progress
        if current_state.get("knowledge_graph_json"):
            graph = KnowledgeGraph.from_json(current_state["knowledge_graph_json"])
            
            print(f"\n📚 Topic: {graph.topic}")
            print(f"   Current Node: {current_state.get('current_node_id', 'N/A')}")
            
            if current_state.get("question"):
                print(f"\n❓ Last Question:")
                print(f"   {current_state['question']}")
            
            if current_state.get("score"):
                print(f"\n📊 Last Score: {current_state['score']:.0f}/100")
                print(f"   Feedback: {current_state.get('feedback', '')[:100]}...")
            
            # Show progress
            summary = graph.get_progress_summary()
            print(f"\n📈 Overall Progress:")
            print(f"   Completed: {summary['completed']}/{summary['total_nodes']} nodes")
            print(f"   In Progress: {summary['in_progress']}")
            print(f"   Average Proficiency: {summary['avg_proficiency']:.1f}%")
            
            # To continue learning, you would:
            # 1. Update student_answer in the state
            # 2. Call app.stream(updated_state, config=config)
            
            print("\n💡 To continue learning:")
            print("   1. Update the 'student_answer' field")
            print("   2. Call app.stream(updated_state, config=config)")
            print("   3. System will grade and ask next question")
            
        else:
            print("\n⚠️  Session exists but no curriculum data found")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_resume_session()
