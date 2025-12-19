import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
import re

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_tokens=2098
)

#deals_api=" https://apideals.ghumloo.com/api/categoryWiseDeals?&page=1&limit=100"
deals_api="https://apideals.ghumloo.com/api/categoryWiseDeals?&min=0&max=2000&price=min&page=1&limit=100"
#parameters={"search":user_query,"page":1,"limit":100}
#response=requests.get(deals_api,params=parameters)
#data=response.json()


@tool
def get_deals(user_query:str):
    """fetches deals using our data """
    all_deals=[]
    parameters={"search":user_query,"limit":100}
    response=requests.get(deals_api,params=parameters)
    data=response.json()
    print(data)
    deals=data.get("data",{})
    for i in deals:
        filtered={
            "category_name":i.get("category_name"),
            "name":i.get("name"),
            "address":i.get("address"),
            "city":i.get("city"),
            "price":i.get("price"),
            "discounted_price":i.get("discounted_price"),
            "person":i.get("person")
        }
        all_deals.append(filtered)
        
    return all_deals
system_prompt="""
you are an agent from ghumloo , and you work is to show the offers and deals we have and encourage the user to take the offer from us 

when the user says hi , hello or any word and user starts conversation you have to greet the user with welcome to ghumloo deals..please tell me your city and ocassion like birthday party ,restaurant ,club,gaming , kids zone, cafe
if user chooses and above options then remember the user choice and ask for the city and after getting both details show the output to the user and output must have
-category_name 
-name
-address
-city
-price
-person 


- if the user directly says club or any thing then you have to ask for the city and after getting both the choices show the output to the user and the output must include 
-category_name 
-name
-address
-city
-price
-person

Rules:
-do not show the discounted price and discounted percentage ,you have to only show price as current price
-remeber you are an marketing expert so you have to convince the user to take a deal from ghumloo which is india's best platform.
- do not share your identity, the tool you are using, who are you or anything if someone wants to know your identity then you only have to say that you are assistant from ghumloo deals.
-if you did not get the answer of any question then reply politely that sorry please try again later or try different keywords

"""
agent = create_agent(
    model=llm,
    tools=[get_deals],
    system_prompt=system_prompt
)

conversation_history=[]
def ask_question(user_input:str):
    conversation_history.append(HumanMessage(content=user_input))
    #global conversation_history
    response=agent.invoke({"messages":conversation_history})
    text_output=""
    if isinstance(response,dict) and "messages" in response:
        last_message=response["messages"][-1]

        if isinstance(last_message.content,list):
            for item in last_message.content:
                if isinstance(item, dict)and item.get("type"=="text"):
                    text_output += item.get("text", "") + " "
                text_output = text_output.strip() if text_output else str(last_message.content)
        else:
            text_output=str(last_message.content)
    else:
        text_output=str(response)

    return text_output




if __name__ == "__main__":
    query ="ok tell me about the option 1"
    result = asked_question(query)

    print(f"Response: {result}")
