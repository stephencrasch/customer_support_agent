"""
Test the simplified adaptive learning system
"""
from agents import StudyState, app

def test_simplified_flow():
    """Test the simplified router-based flow."""
    print("="*70)
    print("SIMPLIFIED ADAPTIVE LEARNING SYSTEM TEST")
    print("="*70)
    
    # Initial state (matches the canonical diagnostic agent schema)
    initial_state: StudyState = {
        "messages": [],
        "user_id": "test_simplified_python",
        "knowledge_graph": "",
        "ultimate_goal": "Python Programming",
        "current_focus": "python_programming",
        "student_answer": "",
        "last_score": 0,
        "awaiting_choice": False,
        "last_tutor_message": "",
        "last_dialogue_update_user_text": "",
        "next": "ask",
    }
    
    print("\n📚 Starting session...")
    print("-"*70)
    
    # Configuration with thread_id
    config = {
        "configurable": {"thread_id": "test_simplified_python"},
        "recursion_limit": 10
    }
    
    # Stream execution
    step = 0
    for state_update in app.stream(initial_state, config=config):
        step += 1
        node_name = list(state_update.keys())[0]
        print(f"\n→ Step {step}: {node_name}")
        
        # Stop after a few steps to show the flow
        if step >= 5:
            break
    
    # Get final state
    final_state = app.get_state(config).values
    
    print("\n" + "="*70)
    print("📊 FINAL STATE")
    print("="*70)
    print(f"Goal: {final_state.get('ultimate_goal')}")
    print(f"Current Focus: {final_state.get('current_focus')}")
    print(f"Next Action: {final_state.get('next')}")
    print(f"Messages: {len(final_state.get('messages', []))} messages")
    
    # Show last message
    if final_state.get('messages'):
        last_msg = final_state['messages'][-1]
        print(f"\n💬 Last Message:")
        print(f"   {last_msg.content[:150]}...")
    
    print("\n✅ Test completed!")
    print("\n🔑 Notes:")
    print("   - This test is a lightweight smoke run for the canonical diagnostic flow.")
    print("   - The graph is intentionally one-assistant-message-per-user-turn.")


if __name__ == "__main__":
    try:
        test_simplified_flow()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
