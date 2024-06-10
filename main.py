#Import Libraries
import os 
from llm_legal_summariser import case_overview


if __name__ == "__main__":

    #This is Allahabad high court case in legal_text.txt file and we will generate its legal summary.
    #Generate your own legal summary by substituting a legal text in legal_text.txt file.

    file_path = 'legal_text.txt'

    #Call case_overview function
    legal_summary = case_overview(file_path)

    #Print the summary in a txt file.
    summary_file_name = 'legal_summary'
    file_name = f"{summary_file_name}.txt"
    file_path = os.path.join(os.getcwd(), file_name)
    
    # Write content to the file
    with open(file_path, 'w') as file:
        file.write(legal_summary)

    
    




