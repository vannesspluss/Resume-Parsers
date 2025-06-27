import os
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader

# Setup API Key
api_key = os.environ["OPENAI_API_KEY"]

# Define schema
class Edu(BaseModel):
    University: str
    Degree: str
    Gpax: Optional[float] = Field(default=None, ge=0, le=10.0)
    Graduation: Optional[int] = Field(default=None)

class Exp(BaseModel):
    Company: Optional[str] = None
    Duration: Optional[str] = None
    Position: Optional[str] = None
    Responsibilities: Optional[List[str]] = None

class Resume(BaseModel):
    Name: str
    Gender: Optional[str] = None
    DOB: Optional[str] = None
    Age: Optional[int] = None
    Email: str
    Phone: str
    Education: Optional[List[Edu]] = None
    Experience: Optional[List[Exp]] = None
    Skills: Optional[List[str]] = None

# Set up LangChain
resume_template = """
You are an AI assistant tasked with extracting structured information from a technical resume.
Only Extract the information that's present in the Resume class.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)
prompt_template = PromptTemplate(template=resume_template, input_variables=["resume_text"])
model = init_chat_model(model="gpt-4o-mini", model_provider="openai").with_structured_output(Resume)

# Main function to extract structured data
def parse_resume(file_path: str) -> Resume:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    resume_text = "\n".join([doc.page_content for doc in docs])

    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)
    return result, resume_text
