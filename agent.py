

import os 
import json
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool


openai_key = json.load(open('credentials.json'))['openai']
os.environ['OPENAI_API_KEY'] = json.load(open('credentials.json'))['openai']


df_carbon = pd.read_csv('Datasets/hourly_42101_2022.zip')
df_brightness = pd.read_csv('Datasets/modis_2022_United_States.csv')

llm = OpenAI(model_name='gpt-4', temperature=0)

stats_agent = create_pandas_dataframe_agent(llm, [df_carbon, df_brightness], verbose=True)


tools = [
    Tool(
        name = "Stats Agent",
        func=stats_agent.run,
        description="Useful when you need to consult statistical data from diferent sources",
        return_direct=True
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,  handle_parsing_errors=True)

#Example
agent.run("what is the average brightness and Carbon monoxide in Califoria?") 


