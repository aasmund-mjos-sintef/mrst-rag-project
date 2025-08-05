# MRST RAG Project

This is a proof-of-concept application meant to demonstrate a
compentece search / guidance tool based on generativ AI using the MRST
database of papers and documentation.

## Getting Started

Create a .env file in the main directory and enter\\
```LANGCHAIN_OPENAI_API_KEY = <your_api_key>```\\
```MRST_REPOSITORY_PATH = <full_path_to_downloaded_mrst_repository>```\\

You can use the .env.example file as an example. Any environment variable associated with LangSmith is not needed,
but can help with debugging openai_api errors or find out where and why the program does something stupid.

It's important to download the original MRST repository if you want to use the github search.
If you don't download the original MRST repository, you will get an error if you include github search in the settings.

You need to be on python version newer or equal to 3.13
Create a virtual environment by running \\
```python3 -m venv .venv``` \\

Activate the environment by running \\
```source .venv/bin/activate``` \\

For this project you need graphviz downloaded on your computer.
On Debian/Ubuntu systems run \\
```apt install graphviz``` \\

you may need to run this instead \\
```sudo apt install graphviz``` \\
and enter your password to get permission

Install the requirements by running \\
```pip install -r requirements.txt``` \\

Prepare the source code package by running
```pip install -e .``` \\

## Running the Program

To run the program, navigate to the frontend folder and enter
```streamlit run app.py```

If everything is set up correctly, you should see
![Example Image](images/app_loaded.png)

## Image of Excecution Graph

![Excecution Graph](images/graph_vizualization.png)

