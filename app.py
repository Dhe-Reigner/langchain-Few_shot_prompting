from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_openai  import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key = api_key)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_template = """
User : {color}
AI : {emotion}
"""

example_prompt = PromptTemplate(
    input_variables = ["color", "emotion"],
    template = example_template
)

prefix = """You are an expert in color psychology who understands how different colors evoke specific human emotions. Below are some examples that show the emotional associations of different colors:
"""

suffix = """
Now, based on this pattern, identify the emotion commonly associated with the following color:

Color: {input}
Emotion:

"""

few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    example_separator = "\n\n",
    input_variables=["input"]
)

formatted_prompt = few_shot_prompt.format(input="purple")

prompt = PromptTemplate(template=formatted_prompt, input_variables=[])

chain = prompt | llm

response = chain.invoke({})

print("Color: purple")
print("Emotion:", response.content)