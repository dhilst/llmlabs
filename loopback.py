import argparse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import time
import sys # Import sys for stdout flushing

def main():
    parser = argparse.ArgumentParser(description="Run a conversational loop between two Ollama LLM models with streaming output.")
    parser.add_argument("--m1", type=str, required=True, help="Name of the first Ollama LLM model (e.g., 'llama2').")
    parser.add_argument("--m2", type=str, required=True, help="Name of the second Ollama LLM model (e.g., 'mistral').")
    parser.add_argument("--prompt", type=str, required=True, help="The initial prompt to start the conversation.")
    parser.add_argument("--max-turns", type=int, default=-1,
                        help="Maximum number of turns in the conversation loop. Use -1 for infinite turns.")

    args = parser.parse_args()

    # Initialize the Ollama LLM models
    print(f"Initializing model 1: {args.m1}")
    model1 = ChatOllama(model=args.m1)
    print(f"Initializing model 2: {args.m2}")
    model2 = ChatOllama(model=args.m2)

    print("\n--- Starting Conversation (Streaming Output) ---")
    print(f"Initial Prompt: {args.prompt}")
    if args.max_turns == -1:
        print("Running in infinite conversation mode. Press Ctrl+C to stop.")
    else:
        print(f"Conversation will run for a maximum of {args.max_turns} turns.")

    current_message = args.prompt
    chat_history = []

    # Agent 1 starts the conversation
    try:
        print(f"\n--- Turn 1 (Agent 1: {args.m1}) ---")
        print(f"Agent 1 ({args.m1}): ", end="", flush=True) # Prepare for streaming output

        # Stream output from model 1
        full_response1 = ""
        for chunk in model1.stream([HumanMessage(content=current_message)]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response1 += chunk.content
        current_message = full_response1
        print() # Newline after agent's full response

        chat_history.append(HumanMessage(content=args.prompt))
        chat_history.append(AIMessage(content=current_message))

        turn_count = 1
        while args.max_turns == -1 or turn_count < args.max_turns:
            turn_count += 1

            if turn_count % 2 == 0:
                # Agent 2's turn
                print(f"\n--- Turn {turn_count} (Agent 2: {args.m2}) ---")
                print(f"Agent 2 ({args.m2}): ", end="", flush=True)

                # Stream output from model 2
                full_response2 = ""
                # Pass the conversation history + current message as input
                for chunk in model2.stream(chat_history + [HumanMessage(content=current_message)]):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response2 += chunk.content
                current_message = full_response2
                print() # Newline after agent's full response
                chat_history.append(AIMessage(content=current_message))
            else:
                # Agent 1's turn
                print(f"\n--- Turn {turn_count} (Agent 1: {args.m1}) ---")
                print(f"Agent 1 ({args.m1}): ", end="", flush=True)

                # Stream output from model 1
                full_response1 = ""
                # Pass the conversation history + current message as input
                for chunk in model1.stream(chat_history + [HumanMessage(content=current_message)]):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response1 += chunk.content
                current_message = full_response1
                print() # Newline after agent's full response
                chat_history.append(AIMessage(content=current_message))

            # Small delay for readability in infinite mode
            if args.max_turns == -1:
                time.sleep(0.5) # Slightly reduced sleep for faster streaming feel

    except KeyboardInterrupt:
        print("\n--- Conversation interrupted by user (Ctrl+C) ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your Ollama server is running and the specified models are downloaded.")
        print("You can download models using: ollama run <model_name>")

    print("\n--- Conversation End ---")

if __name__ == "__main__":
    main()
