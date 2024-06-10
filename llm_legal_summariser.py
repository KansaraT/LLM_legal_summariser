# Import necessary modules from langchain and other libraries
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains import LLMChain
from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
import os 
from langchain_anthropic import AnthropicLLM


# Function to load and split a text file into chunks
def chunks(txt_file_path):
    loader = TextLoader(txt_file_path)  # Load the text file
    docs = loader.load_and_split()      # Split the text file into documents
    return docs


# Function to summarize the content of a PDF file
def summarize_pdf(txt_file_path, map_reduce_chain, text_splitter):
    split_docs = text_splitter.split_documents(chunks(txt_file_path)) # Split the documents
    return map_reduce_chain.run(split_docs)  # Run the map-reduce chain on the split documents


def case_overview(txt_file_path):
    # Set API key for Anthropic LLM
    os.environ["ANTHROPIC_API_KEY"] = 'your_anthropic_api_key'

    # Initialize the Anthropic LLM with the model "claude-instant-1.2"
    llm = AnthropicLLM(model="claude-instant-1.2", max_tokens=2000, temperature=1)

    # Configure the text splitter to split documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Split on double newline
        chunk_size=1000,   # Maximum chunk size
        chunk_overlap=120, # Overlap between chunks
        length_function=len, # Function to measure length
        is_separator_regex=False, # Separator is not a regex
    )

    # Define the mapping prompt template for the map step
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, summarize into meaningful
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Define the reduce prompt template for the reduce step
    reduce_template = """The following is a set of summaries:
    {doc_summaries}
    Take these and distill them into a final consolidated summary with title (mandatory) in bold with important key points. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Create a chain to combine document summaries
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Create a chain to reduce the documents by combining summaries
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=5000,
    )

    # Create the map-reduce chain combining the map and reduce steps
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    result_summary = summarize_pdf(txt_file_path, map_reduce_chain, text_splitter)
    return(result_summary)

