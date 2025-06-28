import argparse
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic_core.core_schema import field_after_validator_function


def create_executor_chain(model_name: str):
    llm = OllamaLLM(
        model=model_name,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    template_str = """You are a helpful assistant. Your task is: {task}

Please complete the task with reasoning, examples, or any useful process.
Output:"""
    prompt = PromptTemplate.from_template(template_str)
    return prompt, llm, StrOutputParser()


def create_formatter_chain(model_name: str):
    llm = OllamaLLM(
        model=model_name,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    template_str = """You are text extrator and format output helpful agent

Your job is to take the following:
Task description:
{task}

Original output:
{raw_output}

{regex_section}

Output instructions:
{output_format}

Now extract the desired answer from the Original unstructured output
and return it in the specified format in Output instructions:

Formatted Output:"""
    prompt = PromptTemplate.from_template(template_str)
    return prompt, llm, StrOutputParser()

def create_improver_formatter_chain(model_name: str):
    llm = OllamaLLM(
        model=model_name,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    template_str = """You are an expert at formatting instructions

Your job is to take the following:
- The original output: {raw_output}
- Output instructions: {output_format}

{regex_section}

Now improve the formatting instructions:

Formatting instructions:"""
    prompt = PromptTemplate.from_template(template_str)
    return prompt, llm, StrOutputParser()


def main():
    parser = argparse.ArgumentParser(description="LangChain Ollama Agent Runner")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama3, mistral, etc.)")
    parser.add_argument("--task", required=True, help="Prompt describing the task to be executed")
    parser.add_argument("--output", required=True, help="Prompt describing the output formatting instructions")
    parser.add_argument("--regex", required=False, help="Regex pattern that the final output must match")
    parser.add_argument("--retries", required=False, type=int, default=10, help="Regex pattern that the final output must match")
    args = parser.parse_args()

    # === EXECUTOR AGENT ===
    print("\n=== [1] Executing Task ===")
    exec_prompt, exec_llm, exec_parser = create_executor_chain(args.model)
    rendered_exec_prompt = exec_prompt.format(task=args.task)
    print("\n--- Prompt sent to LLM (Executor Agent) ---\n")
    print(rendered_exec_prompt)

    raw_output = exec_parser.invoke(exec_llm.invoke(rendered_exec_prompt))

    print("\n\n=== [2] Raw Output Captured ===\n")
    print(raw_output)

    # === FORMATTER AGENT ===
    print("\n=== [3] Formatting Output ===")
    fmt_prompt, fmt_llm, fmt_parser = create_formatter_chain(args.model)

    # === IMPROVER AGENT ===
    print("\n=== [4] Formatting Output ===")
    imp_prompt, imp_llm, imp_parser = create_improver_formatter_chain(args.model)

    instructions = args.output

    regex_instruction = ""
    if args.regex:
        regex_instruction = f"The final output must strictly match the following regular expression:\n{args.regex}\n"

    def format_once(raw_output, instructions):
        return fmt_parser.invoke(fmt_llm.invoke(fmt_prompt.format(
            task=args.task,
            raw_output=raw_output,
            output_format=instructions,
            regex_section=regex_instruction
        )))

    def improve_instructions(raw_output, instructions):
        return imp_parser.invoke(imp_llm.invoke(imp_prompt.format(
            raw_output=raw_output,
            output_format=instructions,
            regex_section=regex_instruction,
        )))

    retries = args.retries
    formatted_output = raw_output
    for attempt in range(1, retries + 1):
        print(f"\n--- Attempt {attempt} ---\n")
        rendered_fmt_prompt = fmt_prompt.format(
            task=args.task,
            raw_output=formatted_output,
            output_format=args.output,
            regex_section=regex_instruction
        )
        print("\n--- Prompt sent to LLM (Formatter Agent) ---\n")
        print(rendered_fmt_prompt)

        formatted_output = format_once(formatted_output, instructions)

        if args.regex:
            print("Regex:", args.regex)
            if re.fullmatch(args.regex, formatted_output.strip(), re.DOTALL):
                print("\n\n=== [4] Final Formatted Output (matched regex) ===\n")
                print(formatted_output)
                break
            else:
                instructions = improve_instructions(raw_output, instructions)
                print("\n❌ Output did not match regex, retrying...\n")
        else:
            print("\n\n=== [4] Final Formatted Output ===\n")
            print(formatted_output)
            break
    else:
        print("\n⚠️ Failed to produce output matching the regex after 3 attempts.\n")
        print("Last output:\n")
        print(formatted_output)


if __name__ == "__main__":
    main()
