import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import Runnable # Still useful for chaining
from pydantic import BaseModel, Field, ValidationError # Keep ValidationError for direct parsing error handling

# Removed: from langchain.output_parsers.retry import RetryOutputParser
from langchain_core.exceptions import OutputParserException 

# --- Pydantic Model for Evaluator Output (unchanged) ---
class EvaluationOutput(BaseModel):
    """Represents the structured output from the evaluator agent."""
    error_score: float = Field(
        ...,
        description="A numerical score indicating the quality of the output. 0 indicates optimal, greater than 0 indicates sub-optimal.",
    )
    feedback: str = Field(
        ...,
        description="Suggestions on how to improve the result, if the error score is greater than 0. If the error score is 0, this field can contain a congratulatory message.",
    )

# --- Existing functions (unchanged) ---

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


def create_executor_chain(
    model_name: str, temperature: float
) -> Tuple[PromptTemplate, OllamaLLM, StrOutputParser]:
    """
    Creates the executor chain components: prompt, LLM, and output parser.
    This chain is responsible for performing the given task.
    """
    llm = create_llm(model_name, temperature)
    template = """You are an expert system. Your task is: {task}

Please perform this task and return your result in the most effective, complete, and concise way possible.

Output:"""
    prompt = PromptTemplate.from_template(template)
    return prompt, llm, StrOutputParser()


# --- MODIFIED create_evaluator_chain: Returns raw string output ---
def create_evaluator_chain(
    model_name: str, temperature: float
) -> Tuple[PromptTemplate, OllamaLLM, StrOutputParser]:
    """
    Creates the evaluator chain components: prompt, LLM, and StrOutputParser.
    This chain assesses the output of the executor and returns raw text.
    """
    llm = create_llm(model_name, temperature)
    template = """You are a task evaluator. Your job is to assess the result of the following task.

Task: {task}
Output: {output}

Evaluate whether the output correctly and optimally solves the task.
- If the output is optimal, you *must* return: ERROR SCORE: 0
- Otherwise, you *must* return: ERROR SCORE: <number greater than 0>, followed by suggestions on how to improve the result.

Evaluation:"""
    
    prompt = PromptTemplate.from_template(template)
    
    return prompt, llm, StrOutputParser()


# --- MODIFIED: create_formatter_chain (no retry logic here) ---
def create_formatter_chain(
    model_name: str, temperature: float
) -> Tuple[PromptTemplate, OllamaLLM, PydanticOutputParser]:
    """
    Creates the formatter chain components: prompt, LLM, and PydanticOutputParser.
    No retry logic is embedded here.
    """
    llm = create_llm(model_name, temperature)
    
    # The parser for the desired structured output
    pydantic_parser = PydanticOutputParser(pydantic_object=EvaluationOutput)

    # Simplified template for formatter (no error_feedback needed)
    template = """You are a highly skilled formatter. Your task is to take the provided raw evaluation text and strictly format it into a JSON object conforming to the following Pydantic schema:

Raw Evaluation Text:
{raw_evaluation_output}

{format_instructions}

Formatted JSON:"""
    
    prompt = PromptTemplate.from_template(template).partial(
        format_instructions=pydantic_parser.get_format_instructions()
    )
    
    return prompt, llm, pydantic_parser


def create_optimizer_chain(
    model_name: str, temperature: float
) -> Tuple[PromptTemplate, OllamaLLM, StrOutputParser]:
    """
    Creates the optimizer chain components: prompt, LLM, and output parser.
    This chain improves task instructions based on evaluation feedback.
    """
    llm = create_llm(model_name, temperature)
    template = """You are an optimizer that improves task instructions.

Task: {task}
Evaluation Output: {evaluation}

Based on the evaluation, improve the task instructions so that an executor agent is more likely to generate an optimal solution.

If the ERROR SCORE is 0, reply with: "✅ The current task instructions are already optimal."

Improved Instructions:"""
    prompt = PromptTemplate.from_template(template)
    return prompt, llm, StrOutputParser()


# --- MODIFIED evaluate_task: No retry loop for formatter ---
def evaluate_task(
    task: str, executor_model: str, evaluator_model: str, formatter_model: str, temperature: float
) -> Tuple[float, str, EvaluationOutput]:
    """
    Executes a task, evaluates its output (raw text), and then formats that output
    into a structured Pydantic model without retries.
    Returns the error score (float), the executor's output, and the structured evaluation result.
    """
    print("\n🚀 [Executor] Invoking the executor ...")
    exec_prompt, exec_llm, exec_parser = create_executor_chain(executor_model, temperature)
    output = (exec_prompt | exec_llm | exec_parser).invoke({"task": task})

    print("\n🧠 [Evaluator] Invoking the evaluator (generating raw text) ...")
    eval_prompt, eval_llm, eval_parser = create_evaluator_chain(evaluator_model, temperature)
    raw_evaluation_output = (eval_prompt | eval_llm | eval_parser).invoke({"task": task, "output": output})
    
    print("\n✨ [Formatter] Invoking the formatter (no retries) ...")
    format_prompt, format_llm, pydantic_parser_base = create_formatter_chain(formatter_model, temperature)

    # Print the prompt sent to the formatter
    rendered_formatter_prompt = format_prompt.format(raw_evaluation_output=raw_evaluation_output)
    print(f"\n--- Formatter Agent Prompt ---")
    print(rendered_formatter_prompt)
    print(f"--- End Formatter Agent Prompt ---\n")

    try:
        # Chain the formatter components directly
        formatter_chain = format_prompt | format_llm | pydantic_parser_base
        evaluation_result: EvaluationOutput = formatter_chain.invoke({"raw_evaluation_output": raw_evaluation_output})
        print("✅ Formatter output parsed successfully.")
    except (ValidationError, OutputParserException) as e:
        print(f"❌ Failed to parse formatter output: {e}", file=sys.stderr)
        # Return a high error score to indicate parsing failure
        return float('inf'), output, EvaluationOutput(error_score=float('inf'), feedback=f"Formatter failed to parse output: {e}. Raw: {raw_evaluation_output[:200]}...")
    except Exception as e:
        print(f"An unexpected error occurred during formatting: {e}", file=sys.stderr)
        return float('inf'), output, EvaluationOutput(error_score=float('inf'), feedback=f"Unexpected error during formatting: {e}. Raw: {raw_evaluation_output[:200]}...")

    error_score = evaluation_result.error_score
    return error_score, output, evaluation_result


def optimize_task(
    task: str, evaluation: EvaluationOutput, optimizer_model: str, temperature: float
) -> str:
    """
    Optimizes the task instructions based on the evaluation.
    Note: The evaluation parameter now expects a Pydantic EvaluationOutput object.
    """
    opt_prompt, opt_llm, opt_parser = create_optimizer_chain(optimizer_model, temperature)
    # When sending the evaluation to the optimizer, we convert the Pydantic object to a JSON string
    improved_instructions = (opt_prompt | opt_llm | opt_parser).invoke({"task": task, "evaluation": evaluation.model_dump_json()})
    return improved_instructions.strip()


def main():
    """
    Main function to run the instruction optimization loop.
    """
    parser = argparse.ArgumentParser(description="Instruction optimizer loop")
    parser.add_argument("--model", required=True, help="Default model for all agents")
    parser.add_argument("--executor-model", help="Executor model name")
    parser.add_argument("--evaluator-model", help="Evaluator model name")
    parser.add_argument("--optimizer-model", help="Optimizer model name")
    parser.add_argument("--formatter-model", help="Formatter model name (defaults to --model)")
    parser.add_argument("--task", required=True, help="Initial task instructions")
    parser.add_argument("--retries", type=int, default=3, help="Max optimization iterations for the loop")
    # Removed: parser.add_argument("--evaluator-retries", type=int, default=2, help="Max retries for formatter JSON parsing")
    parser.add_argument("--temp", type=float, default=0.8, help="LLM temperature (default=0.8)")

    args = parser.parse_args()

    executor_model = args.executor_model or args.model
    evaluator_model = args.evaluator_model or args.model
    optimizer_model = args.optimizer_model or args.model
    formatter_model = args.formatter_model or args.model

    current_task = args.task
    best_score = float("inf")
    final_output = "" # Store the output from the best performing task

    print(f"Starting optimization for task: '{current_task}' with up to {args.retries} retries.")

    for i in range(args.retries):
        print(f"\n--- 🔁 Iteration {i+1}: Evaluating current task ---\n")
        
        # Removed evaluator_retries from call
        score, output, evaluation_obj = evaluate_task(
            current_task, executor_model, evaluator_model, formatter_model, args.temp
        )
        final_output = output # Keep track of the latest output

        print(f"\n--- Iteration {i+1} Results ---")
        print(f"📉 Error Score: {score}")
        print(f"📝 Task: {current_task}")
        print(f"📤 Output: {output}")
        print(f"📋 Evaluation: {evaluation_obj.model_dump_json(indent=2)}")
        print("------------------------------")

        if score == 0.0:
            print("\n✅ Task instructions are already optimal. Stopping optimization.")
            best_score = 0.0
            break
        elif score == float('inf'): # This will now trigger on first parsing failure from formatter
            print("\n🚨 Formatter failed to produce valid output. Cannot optimize this iteration. Stopping loop.")
            best_score = float('inf')
            break


        print("\n🧠 [Optimizer] Optimizing the prompt instructions ...")
        optimized_instructions = optimize_task(
            current_task, evaluation_obj, optimizer_model, args.temp
        )

        print(f"\n✨ Optimized Instructions:\n{optimized_instructions}\n")
        
        print("\n--- Re-evaluating with optimized instructions ---")
        # Removed evaluator_retries from call
        new_score, new_output, new_evaluation_obj = evaluate_task(
            optimized_instructions, executor_model, evaluator_model, formatter_model, args.temp
        )
        
        print(f"\n🔍 Comparing Scores — Original: {score}, Optimized: {new_score}")
        if new_score < score:
            print("✅ Optimized instructions improved the error score. Using in next round.")
            current_task = optimized_instructions
            best_score = new_score
            final_output = new_output
        else:
            print("🟡 Optimized instructions did not improve error score or made it worse. Keeping previous task instructions.")

    print("\n--- 🎯 Optimization Loop Finished ---")
    print("\n🎯 Final Instructions:")
    print(current_task)
    print("\nFinal Output for Optimal Task:")
    print(final_output)
    print("------------------------------------")
    
    sys.exit(0 if best_score == 0.0 else 1)


if __name__ == "__main__":
    main()
