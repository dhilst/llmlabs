# rpg.py
import argparse
import random
import sys
from typing import Tuple, Dict, List

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field, ValidationError

# --- Pydantic Models for Structured Output ---

class PlayerAttributes(BaseModel):
    """Represents the player's attributes and inventory."""
    life: int = Field(..., description="Current life points, from 0 to 100.")
    strength: int = Field(..., description="Strength score (1-20). Affects physical power.")
    intelligence: int = Field(..., description="Intelligence score (1-20). Affects problem-solving and knowledge.")
    dexterity: int = Field(..., description="Dexterity score (1-20). Affects agility and reflexes.")
    health: int = Field(..., description="Maximum health points.")
    inventory: List[str] = Field(..., description="List of items the player is carrying.")

# --- LLM and Agent Creation Functions ---

def create_llm(model_name: str, temperature: float = 0.8) -> OllamaLLM:
    """
    Initializes and returns an OllamaLLM instance with streaming enabled.
    """
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

def create_master_agent(model_name: str, temperature: float) -> OllamaLLM:
    """
    Creates the Master (Game Master) agent LLM.
    """
    return create_llm(model_name, temperature)

# --- Game Logic Functions ---

def initialize_game(
    master_llm: OllamaLLM, theme: str
) -> Tuple[str, str, PlayerAttributes, str]:
    """
    Initializes the RPG game by having the Master set the story, the human Player create a character,
    the Master generate attributes, and then set the first scene.
    """
    print("\n--- Game Initialization ---")
    
    # 1. Master gives the dungeon story based on the theme
    master_story_prompt_template = PromptTemplate.from_template(
        """You are the Dungeon Master for a text-based RPG. Your goal is to tell a compelling story and guide the player.
The game theme is '{theme}'.

Begin by setting the scene for the dungeon story based on the '{theme}' theme. Make it intriguing.
Story Introduction:"""
    )
    master_story_chain = master_story_prompt_template | master_llm | StrOutputParser()
    
    print(f"\n🌍 DM: Welcome, adventurer! Our tale begins in a {theme} setting.")
    initial_story = master_story_chain.invoke({"theme": theme})
    print(f"DM: {initial_story}\n")

    # 2. Player writes their character (human input)
    print("👤 You are the player. The Dungeon Master has set the stage.")
    player_character_desc = input("Describe your character (name, appearance, background, etc.): ")
    print(f"\nPlayer: {player_character_desc}\n")

    # 3. Master answers with player attributes
    master_attr_parser = PydanticOutputParser(pydantic_object=PlayerAttributes)
    master_attr_prompt_template = PromptTemplate.from_template(
        """You are the Dungeon Master. The player has described their character.
Based on the following character description, generate their attributes and inventory as a JSON object.
Ensure the 'life' and 'health' are between 0 and 100, and 'strength', 'intelligence', 'dexterity' are between 1 and 20.

Character Description:
{character_description}

{format_instructions}

Attributes JSON:"""
    ).partial(format_instructions=master_attr_parser.get_format_instructions())
    
    master_attr_chain = master_attr_prompt_template | master_llm | master_attr_parser
    
    print("DM: Interesting character! Let's see your starting attributes...")
    try:
        player_attrs: PlayerAttributes = master_attr_chain.invoke({"character_description": player_character_desc})
        print(f"DM: Here are your starting attributes:")
        print(f"   Life: {player_attrs.life}/{player_attrs.health}")
        print(f"   Strength: {player_attrs.strength}")
        print(f"   Intelligence: {player_attrs.intelligence}")
        print(f"   Dexterity: {player_attrs.dexterity}")
        print(f"   Inventory: {', '.join(player_attrs.inventory) if player_attrs.inventory else 'None'}\n")
    except ValidationError as e:
        print(f"DM: I'm having trouble understanding your character's stats. Let's use defaults. Error: {e}", file=sys.stderr)
        # Fallback to default attributes if parsing fails
        player_attrs = PlayerAttributes(
            life=50, strength=10, intelligence=10, dexterity=10, health=50, inventory=["basic sword", "small pouch"]
        )
        print(f"DM: For now, we'll use default attributes:")
        print(f"   Life: {player_attrs.life}/{player_attrs.health}")
        print(f"   Strength: {player_attrs.strength}")
        print(f"   Intelligence: {player_attrs.intelligence}")
        print(f"   Dexterity: {player_attrs.dexterity}")
        print(f"   Inventory: {', '.join(player_attrs.inventory)}\n")


    # 4. Master tells the first scene
    master_first_scene_prompt_template = PromptTemplate.from_template(
        """You are the Dungeon Master. The game has begun.
The story so far: {initial_story}
Player Character: {character_description}
Player Attributes: {player_attributes}

Describe the first immediate scene for the player. What do they see? What is their current predicament or opportunity?
End by asking "What do you do?"

First Scene:"""
    )
    master_first_scene_chain = master_first_scene_prompt_template | master_llm | StrOutputParser()

    first_scene = master_first_scene_chain.invoke({
        "initial_story": initial_story,
        "character_description": player_character_desc,
        "player_attributes": player_attrs.model_dump_json(indent=2) # Pass JSON for context
    })
    print(f"DM: {first_scene}\n")

    return initial_story, player_character_desc, player_attrs, first_scene

def play_round(
    master_llm: OllamaLLM,
    game_history: List[Dict[str, str]], # To pass full conversation context
    player_character_desc: str,
    player_attrs: PlayerAttributes,
    current_scenario: str,
) -> Tuple[str, str]:
    """
    Plays a single round of the RPG game with human player input.
    The Master describes the scenario and asks for player action.
    The human Player responds with their action via keyboard input.
    """
    print(f"--- Round ---")

    # 1. Master describes the scenario and asks for player action
    master_turn_prompt_template = PromptTemplate.from_template(
        """You are the Dungeon Master. Continue the story.
Player Character: {character_description}
Player Attributes: {player_attributes}
Game History (previous turns):
{game_history_str}

Current Scenario: {current_scenario}

Describe the current situation, what the player sees, and ask for their next action.
Scenario Description and Question:"""
    )

    # Format game history for prompt
    game_history_str = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in game_history])

    master_turn_chain = master_turn_prompt_template | master_llm | StrOutputParser()
    
    master_description_and_question = master_turn_chain.invoke({
        "character_description": player_character_desc,
        "player_attributes": player_attrs.model_dump_json(indent=2),
        "game_history_str": game_history_str,
        "current_scenario": current_scenario
    })
    print(f"DM: {master_description_and_question}\n")

    # 2. Human Player tells what they want to do (keyboard input)
    player_action = input("Your Action: ")
    print(f"Player: {player_action}\n")

    # For now, we'll let the Master's next turn implicitly process the action.
    # In a more complex game, you'd add logic here to update player_attrs based on action.
    
    return master_description_and_question, player_action # Return master's last output and player's action for next round's context

def main():
    parser = argparse.ArgumentParser(description="Text-based RPG game between Master agent and Human Player.")
    parser.add_argument("--model", required=True, help="Default model for the Master agent (e.g., 'ollama/llama2')")
    parser.add_argument("--master-model", help="Specific model for the Master agent (overrides --model)")
    parser.add_argument("--theme", choices=["zombie apocalypse", "sci-fi", "fantasy"],
                        help="Choose a game theme. If not specified, one will be chosen randomly.")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM temperature for the Master (default=0.7)")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum number of game rounds to play.")

    args = parser.parse_args()

    master_model = args.master_model or args.model
    chosen_theme = args.theme if args.theme else random.choice(["zombie apocalypse", "sci-fi", "fantasy"])

    print(f"Setting up RPG with theme: {chosen_theme.upper()}")
    print(f"Master Model: {master_model}, Temperature: {args.temp}")

    # Initialize Master LLM
    master_llm = create_master_agent(master_model, args.temp)

    # Game Initialization
    initial_story, player_character_desc, player_attrs, first_scene = initialize_game(
        master_llm, chosen_theme
    )

    current_scenario = first_scene
    game_history = [] # Stores (speaker, message) for conversation context

    print("\n--- Game Starts! ---")

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n=== ROUND {round_num} ===")

        # Add current scenario to history before master's next turn
        if round_num == 1:
             game_history.append({"speaker": "DM (Initial Scene)", "message": current_scenario})
        else:
             # In subsequent rounds, current_scenario holds the DM's last description
             game_history.append({"speaker": "DM", "message": current_scenario})


        master_output_this_round, player_action_this_round = play_round(
            master_llm, game_history, player_character_desc, player_attrs, current_scenario
        )
        
        # Update current scenario for the next round based on Master's latest output
        current_scenario = master_output_this_round

        # Add player action to history for next round's context
        game_history.append({"speaker": "Player", "message": player_action_this_round})


        # Example: Simple game ending conditions (you can expand this significantly)
        if "victory" in master_output_this_round.lower():
            print("\n🎉 DM: Congratulations, adventurer! You have achieved victory!")
            break
        if player_attrs.life <= 0: # Basic health check
            print("\n💀 DM: Alas, your journey ends here. You have fallen.")
            break
        
        # Allow player to quit
        if player_action_this_round.lower().strip() in ["quit", "exit", "stop"]:
            print("\nGame ended by player.")
            break

    print("\n--- Game Over ---")
    sys.exit(0)

if __name__ == "__main__":
    main()
