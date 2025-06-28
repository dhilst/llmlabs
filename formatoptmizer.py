"""
Prompt format optimizer

Idea: 
- You have a task to be done with LLMs
- You need whatever model to output in a specific format
- You write output formating specification by hand
- It does not work very well with most of the models
- You then have this idea:
  - Let's get the initial output format instructions
  - And ask a model to generate 10 variations of the output instructions
  - And ask the same model to try these 10 variations and output
    a drift value which is how distant the output is from the desired output
  - Then select the 5 variations that have smaller drift:
    - Use elitism to preserve the first one
    - Use crossover between 2nd and 3th to get the new 2nd and 4th and 5th the new 3rd
    - Use mutation over 2nd and 3rd to get the 5th 
  - Run for N generations where N is a parameter, print drift min, avg, max over time

Example Prompt:

You're an adventurer in a maze. Use commands to interact with
the maze.

Commands:
- n, s, e, w:  moves to north, south, east and west respectively
- eat: eat food from inventory
- drink: drink from cantil
- attack: perform an attack
- defend: perform a defence
- look: see around

Environment:
You see two rats in front of you

What you do?

------


Pipeline

Runtime:
-----------
TASK(instructions, environment) -> runner -> UNSTRUCTURED OUTPUT 
UNSTRUCTURED OUTPUT -> interpreter (formating instructions) -> STRUCTURED OUTPUT


Output Optimizer:
TASK(instructions, environment) -> runner -> UNSTRUCTURED OUTPUT 

UNSTRUCTURED OUTPUT -> interpreter (formating instructions) -> STRUCTURED OUTPUT

STRUCTURED OUTPUT -> evaluator (formating instructions) -> DRIFT

formating instructions -> format instructions (generator, N) -> [formating instructions + variation, ...]

[formating instructions + variation, ...] -> evaluator -> scored instructions

select best 5 instruction set









"""

type UnstructuredOutput = str
type StructuredOutput = str
type DriftOutput = str

def runner(
    task_desc: str,
) -> UnstructuredOutput:

    return f"""
You're TASK RUNNER

I'm going to give you a task, you're going
to evaluate it, analyse it, write a plan
and execute it

TASK DESCRIPTION:
{task_desc}
"""

def formatter(
    task_desc: str, 
    unstructured_output: str,
    instructions: str,
) -> StructuredOutput:
    return f"""
You're UNSTRUCTURED OUTPUT INTERPRETER

I'm going to give you a task description, the
RUNNER agent is an LLM agent that will execute
that task and produce UNSTRUCTURED OUTPUT.

I'm going to give you a list of OUTPUT FORMAT INSTRUCTIONS
and the UNSTRUCTURED OUTPUT.

Your task is to interpret the UNSTRUCTURED OUTPUT and generate
STRUCTURED OUTPUT, following the OUTPUT FORMAT INSTRUCTIONS

TASK DESCRIPTION:
{task_desc}

UNSTRUCTURED OUTPUT:
{unstructured_output}

OUTPUT FORMAT INSTRUCTIONS:
{instructions}

STRUCTURED OUTPUT:
"""

def evaluator(
    task_desc: str, 
    unstructured_output: str,
    instructions: str,
    structured_output: str,
) -> str:
    return f"""
You're OUTPUT EVALUATOR

I'm going to give you a list of TASK DESCRIPTION, OUTPUT INSTRUCTIONS and a
STRUCTURED OUTPUT. Your task is to output a DRIFT parameter that tells how
distant the STRUCTURED OUTPUT is from completely satsifying the OUTPUT
INSTRUCTIONS. This value must
range from 0 to positive infinity where

* 0: Perfect match, the output is valid 
* N > 0: The hypothetical distance from the output format


TASK DESCRIPTION:
{task_desc}

UNSTRUCTURED OUTPUT:
{unstructured_output}

OUTPUT FORMAT INSTRUCTIONS:
{instructions}

STRUCTURED OUTPUT:
{structured_output}

DRIFT:
"""
