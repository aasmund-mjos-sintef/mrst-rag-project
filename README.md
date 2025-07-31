# MRST RAG Project

This is a proof-of-concept application meant to demonstrate a
compentece search / guidance tool based on generativ AI using the MRST
database of papers and documentation.

## Getting started

Create a .env file in the main directory and enter
```OPENAI_API_KEY = <your_api_key>``` and
```LANGCHAIN_OPENAI_API_KEY = <your_api_key>```
to send api-requests to openai and langchain openai. 

Create a virtual environment and make sure to run
```pip install -r requirements.txt```

To run the program, navigate to the frontend folder and enter
```streamlit run app.py```