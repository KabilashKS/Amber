from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from config import master_llm
from fpdf import FPDF
from crewai.tools import tool

load_dotenv()

@tool("PDF Generator")
def generate_pdf(content: str) -> str:
    """Use this tool to save the final research report as a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    safe_content = content.encode('latin-1','replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_content)
    pdf.output('Final_Research_Report.pdf')
    return "Pdf successfully created and saved!"

publisher_agent = Agent(
    role = "Content publisher and pdf generation",
    goal="Take the final human-approved draft and save it as a PDF.",
    backstory="""You are the final step of the pipeline. You take the text that the human approved, and you use your PDF Generator tool to save it to the computer.",
""",
    tools = [generate_pdf],  
    llm=master_llm
)