import json
from dotenv import load_dotenv 

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import streamlit as st
import traceback
import pandas as pd 
from langchain.callbacks import get_openai_callback 
from util import  get_table_data, RESPONSE_JSON , get_table_data_ans
from langchain_huggingface import HuggingFaceEndpoint
import os 
load_dotenv()
hf_key = os.environ["HF_token"] 
import subprocess

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id = repo_id , temperature=0.1 , token = hf_key)


template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions for students to help in preparing for his exam from the text . 
Make sure that questions are not repeated and check all the question to be conforming to the text as well.
Make sure to format you response like the RESPONSE_JSON below and use it as a guide \
Ensure to make the {number} MCQs.
### RESPONSE_JSON 
{response_json}
"""

quiz_prompt = PromptTemplate(
    input_variables = ["text" , "number" , "response_json"],
    template = template
)

quiz_chain = LLMChain(
    llm = llm , prompt = quiz_prompt , output_key = "quiz" , verbose = True
)
stemplate = """
Text: {text}
You are an expert summary generator and english grammarian, create a well described summary for students to prepare from {text} for their exam\
make sure all important points are retained. should be in text format , paragraph 
"""
sum_prompt = PromptTemplate(
    input_variables = ["text" ],
    template = stemplate
)
summ_chain = LLMChain(
    llm = llm , prompt = quiz_prompt , output_key = "sum" , verbose = True
)


template = """You are an expert english grammarian and counter. Given a multiple choice quiz for students to help in preparing for his exam from the text\
You need to check if the questions make sense and are not repeated. Also make sure there are {number} number of MCQs
If quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and generate questions if there are less than {number}. 
Quiz MCQs:
{quiz}
Return the quiz in format {response_json} :"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=[ "number","quiz"], template=template
)
review_chain = LLMChain(
    llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True
)

generate_evaluate_chain = SequentialChain(
    chains = [summ_chain,quiz_chain , review_chain] , 
    input_variables = ["text" , "number" , "response_json"],
    output_variables  = ["quiz" , "review" , "sum"], 
    verbose = True
)

st.title("QUIZZBOT")

with st.form("user_inputs") :
    Text = st.text_input("Insert the text : ")
    mcq_count = st.number_input("No. of Questions : " , min_value=1 , max_value=10)
    button = st.form_submit_button("QUIZ TIME")
    


if button and Text and mcq_count : 
    with st.spinner("Generating Questions ...") : 
        with get_openai_callback() as cb:
            response = generate_evaluate_chain(
                {
                    "text" : Text ,
                    "number" : mcq_count,
                    "response_json" : json.dumps(RESPONSE_JSON),
                }
                
            )
    
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

        if isinstance(response, dict):
            # Extract quiz data from the response
            quiz = response.get("review", None)
            if quiz is not None:
                table_data = get_table_data(quiz)
                if table_data is not None:
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                   
                    st.table(df)
                    
                    dfans = pd.DataFrame(get_table_data_ans(quiz))
                    dbut = st.download_button("Download" , data = dfans.to_csv().encode("utf-8") , mime="text/csv" , file_name="Solutions.csv")
                else:
                    st.error("Error in table data")
        else:
            st.write(response)
