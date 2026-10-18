import os
from dotenv import load_dotenv
from config import master_llm
from crewai import Agent
from crewai.tools import tool
from crewai_tools import SerperDevTool

load_dotenv()
_serper = SerperDevTool()

@tool("search_web")
def search_web(query: str) -> str:
    """Search Google and academic sources for authoritative information, published studies, and direct URLs."""
    try:
        raw_res = _serper.run(search_query=query)
        if isinstance(raw_res, dict):
            organic = raw_res.get("organic", [])
            formatted = []
            for item in organic[:6]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                formatted.append(f"- Title: {title}\n  URL: {link}\n  Snippet: {snippet}")
            return "\n\n".join(formatted) if formatted else "No organic results found."
        return str(raw_res)
    except Exception as e:
        return f"Search query failed with error: {e}"

search_tool = search_web

planning_agent = Agent(
    role="Senior Investigative Web Researcher",
    goal="Conduct exhaustive, fact-based internet research to extract, verify, and synthesize highly accurate information regarding the user's query.",
    backstory="""You are an elite web researcher and a meticulous fact-checker. Your primary purpose is to uncover the absolute truth, 
    separate facts from misinformation, and organize complex data into clear, actionable insights. You have a zero-tolerance policy for fake news and unreliable sources.
    Always record the source URL from your search_web tool for every key finding or base research paper found. 
    Never fabricate URLs — only output real URLs returned by your search_web tool.
    CRITICAL INSTRUCTIONS TO FOLLOW:
    1. TOOL USAGE: Call the search_web tool with a clear search query to retrieve real sources and papers.
    2. FACT VERIFICATION: Cross-reference findings and prioritize credible, authoritative sources.
    3. NO FAKE LINKS: Never guess or invent links. Only cite the real URLs provided in the search results.
    4. COMPREHENSIVE OUTPUT: Synthesize your research into detailed, factual notes including direct paper links for the team.""",
    tools=[search_web],
    llm=master_llm,
    max_iter=3
)



