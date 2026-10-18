from planner import planning_agent
from verifier import evidence_agent  
from report import report_agent
from publisher import publisher_agent, generate_pdf
from crewai import Task, Crew
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

class ResearchRequest(BaseModel):
    user_topic:str

Research_Agent = APIRouter(prefix="/api/research", tags=["Research_AI"])
@Research_Agent.post("/api/research")

def do_research(request: ResearchRequest):

    while True:
        # 1. Ask the user for input
        user_topic = input("\nWhat would you like me to research? (Type 'q' to quit): ")
        
        if user_topic.lower() == 'q':
            print("Goodbye!")
            break

        # 2. Insert the user's input into the Task descriptions using f-strings!
        task1 = Task(
            description=f"Search the internet and find accurate information about this topic: {user_topic}",
            expected_output="A list of bullet points containing accurate facts and data.",
            agent=planning_agent
        )

        task2 = Task(
            description=f"Read the research data about {user_topic} and find the relavant evidence to prove why it is better or real.",
            expected_output="Get the relavant evidence for the planning agent to prove how they better and successed.",
            agent=evidence_agent
        )
        
        task3 = Task(
            description=f"Make a report about {user_topic} and combine the planner research findings collected truth evidence about the topic, write a final report with detailed comparsion with planning agent generated answer to satisfy why it is better.",
            expected_output="A written a 1000 words paragraph report from the planner and evidence agent." \
            "This is the format " \
                    " tile"
                "sub-title"
            "paragrapgh or bullet points"
            "below that add the evidence prove",
            agent=report_agent,
            human_input=True
        )

        task4 = Task(
            description="Take the exact report from Task 3 and use your PDF Generator tool to save it.",
            expected_output="A confirmation string that the PDF was saved.",
            agent=publisher_agent,
            context=[task3]
        )

        # 3. Create the crew for this specific job
        my_crew = Crew(
            agents=[planning_agent, evidence_agent, report_agent, publisher_agent],
            tasks=[task1, task2, task3, task4],
            verbose=True
        )

        # 4. Press GO!
        print(f"\n⚡ Sending the team to research: {user_topic}...\n")
        result = my_crew.kickoff()
        