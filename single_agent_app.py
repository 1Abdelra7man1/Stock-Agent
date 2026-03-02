import streamlit as st
from dotenv import load_dotenv
import os

from typing import Annotated
from typing_extensions import TypedDict
import wikipedia
import pandas as pd

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

@tool
def wikipedia_tool(query: Annotated[str, "The Wikipedia search to execute to find key summary information."]):
    
    """Use this to search Wikipedia for factual information."""
    try:
        # Step 1: Search using query
        results = wikipedia.search(query)
        
        if not results:
            return "No results found on Wikipedia."
        
        # Step 2: Retrieve page title
        title = results[0]

        # Step 3: Fetch summary
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"




@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of days of stock data required to respond to the user query."]
) -> str:
    """
    Use this to look-up stock performance data for companies to retrieve a table from a CSV. You may need 
    to convert company names into ticker symbols to call this function, e.g, Apple Inc. -> AAPL, and you may 
    need to convert weeks, months, and years, into days.
    """
    
    # Load the CSV for the company requested
    file_path = f"data/{company_ticker}.csv"

    if os.path.exists(file_path) is False:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Ensure the index is in date format
    stock_df.index = stock_df.index.date
    
    # Maximum num_days supported by the dataset
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    # Get the most recent date in the DataFrame
    final_date = stock_df.index.max()

    # Filter the DataFrame to get the last num_days of stock data
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]


    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\n\n{filtered_df.to_markdown()}"




from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. The chart should be displayed using `plt.show()`.
    """

    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed the Python REPL tool.\n\nPython code executed:\n\`\`\`python\n{code}\n\`\`\`\n\nCode output:\n\`\`\`\n{result}\`\`\`"



class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder= StateGraph(State)

# Add three tools to the list: wikipedia_tool, stock_data_tool, and python_repl_tool
tools= [wikipedia_tool, stock_data_tool, python_repl_tool]

load_dotenv()
groq_key= os.getenv("GROQ_KEY")

llm= ChatGroq(model= "openai/gpt-oss-120b", api_key= groq_key)
llm_with_tools= llm.bind_tools(tools)

def llm_node(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

tool_node= ToolNode(tools)


graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START,"llm")
graph_builder.add_conditional_edges("llm", tools_condition, ["tools", END])
graph_builder.add_edge("tools", "llm")

config= {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer= InMemorySaver()

graph= graph_builder.compile(checkpointer=checkpointer)

# ==============================
# Chat Interface
# ==============================
st.title("🤖 Research & Analysis System")

user_input = st.chat_input("Ask me anything!")

model_response= graph.invoke({"messages": [{"role":"user", "content": user_input}]}, config)
clear_response= model_response["messages"][-1].content

if "messages" not in st.session_state:
    st.session_state.messages= []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])



if user_input:
  
  st.chat_message("user").markdown(user_input)
  st.session_state.messages.append({"role": "user", "content": user_input})


  st.chat_message("ai").markdown(clear_response)
  st.session_state.messages.append({"role": "ai", "content": clear_response})