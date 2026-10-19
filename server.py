import threading
import ctypes
import queue
import uuid
import os
import requests
import json
import re
import time
import multiprocessing as mp
from fpdf import FPDF
from planner import planning_agent
from verifier import evidence_agent  
from report import report_agent
from publisher import publisher_agent, generate_pdf
from crewai import Task, Crew
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import Depends
from sqlmodel import SQLModel, Session, select
from database import engine
from model import Conversation, Message, Project, ProjectConversation
from config import get_llm, AVAILABLE_MODELS
from crewai.tasks.task_output import TaskOutput

# Bulletproof guard for CrewAI TaskOutput: ensure raw is always string-validated
_orig_task_output_init = TaskOutput.__init__
def _safe_task_output_init(self, *args, **kwargs):
    if "raw" in kwargs and not isinstance(kwargs["raw"], str):
        val = kwargs["raw"]
        if isinstance(val, list):
            parts = []
            for item in val:
                if hasattr(item, "arguments"):
                    parts.append(str(item.arguments))
                elif hasattr(item, "function") and hasattr(item.function, "arguments"):
                    parts.append(str(item.function.arguments))
                else:
                    parts.append(str(item))
            kwargs["raw"] = "\n".join(parts) if parts else ""
        else:
            kwargs["raw"] = str(val)
    _orig_task_output_init(self, *args, **kwargs)
TaskOutput.__init__ = _safe_task_output_init


Research_AI = APIRouter(prefix="/api/research", tags=["Research_AI"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_FILE_PATH = os.path.join(BASE_DIR, "amber_logo_transparent.png")
FALLBACK_LOGO = "/home/kabilash/.gemini/antigravity/brain/83399c15-8e53-4883-bc19-86fc1b252c71/amber_logo_transparent.png"

@Research_AI.get("/logo")
def get_amber_logo():
    if os.path.exists(LOGO_FILE_PATH):
        return FileResponse(LOGO_FILE_PATH, media_type="image/png")
    if os.path.exists(FALLBACK_LOGO):
        return FileResponse(FALLBACK_LOGO, media_type="image/png")
    return {"status": "Logo not found"}

active_tasks: dict[str, mp.Process] = {}
active_streaming_tasks: dict[str, bool] = {}

def kill_process(proc: mp.Process):
    """Instantly terminate and kill worker process."""
    if not proc or not proc.is_alive():
        return
    try:
        proc.terminate()
        proc.join(timeout=0.2)
        if proc.is_alive():
            proc.kill()
    except Exception:
        pass

class TerminateRequest(BaseModel):
    task_id: str

class ApproveReportRequest(BaseModel):
    report_text: str
    conversation_id: int | None = None

class ProjectCreateRequest(BaseModel):
    name: str
    instructions: str | None = ""

class ProjectAssignConversationRequest(BaseModel):
    conversation_id: int

class ResearchRequest(BaseModel):
    user_topic: str
    conversation_id: int | None = None
    project_id: int | None = None
    task_id: str | None = None
    model_id: str | None = "nvidia_llama"

@Research_AI.get("/models")
def get_available_models():
    """Returns the list of selectable models for the UI dropdown."""
    return [
        {
            "id": model_id,
            "name": cfg["name"],
            "provider": cfg["provider"],
            "tier": cfg.get("tier", "Fast"),
            "context": cfg.get("context", "")
        }
        for model_id, cfg in AVAILABLE_MODELS.items()
    ]

def get_session():
    with Session(engine) as session:
        yield session

def execute_python_search(topic: str) -> str:
    """Fast, direct Python search via Serper API without LLM middleman (<0.8s)."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "No SERPER_API_KEY found in environment."
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = json.dumps({"q": topic, "num": 8})
        resp = requests.post(url, headers=headers, data=payload, timeout=6)
        if resp.ok:
            data = resp.json()
            organic = data.get("organic", [])
            formatted = []
            for item in organic[:8]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                formatted.append(f"- **Title:** {title}\n  **URL:** {link}\n  **Findings:** {snippet}")
            return "\n\n".join(formatted) if formatted else "No web results found."
        return f"Search API error status: {resp.status_code}"
    except Exception as e:
        return f"Direct search error: {e}"

def _run_crew_synthesis(task_description: str, model_id: str, queue_out):
    """Isolated process worker that executes CrewAI synthesis and communicates result via Queue."""
    try:
        from report import report_agent
        from crewai import Task, Crew
        from config import get_llm

        selected_llm = get_llm(model_id)
        report_agent.llm = selected_llm
        synthesis_task = Task(
            description=task_description,
            expected_output="An expansive, beautifully formatted, comprehensive markdown research report following the exact 7-section structure.",
            agent=report_agent,
            human_input=False
        )
        crew = Crew(
            agents=[report_agent],
            tasks=[synthesis_task],
            verbose=True
        )
        result = crew.kickoff()
        queue_out.put(("success", str(result)))
    except Exception as e:
        queue_out.put(("error", str(e)))

def sanitize_latin1(text: str) -> str:
    """Safely converts unicode characters and emojis to ASCII for classic FPDF."""
    replacements = {
        '\u2014': '--',
        '\u2013': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2022': '-',
        '\u2026': '...',
        '\u00a0': ' ',
        '\u2192': '->',
        '\u2190': '<-',
        '\u2713': '[v]',
        '\u2705': '[v]',
        '\u274c': '[x]',
        '\U0001f4c4': '[Doc]',
        '\U0001f4da': '[Sources]',
        '\U0001f4a1': '[Idea]',
        '\U0001f52c': '[Analysis]',
        '\U0001f4ca': '[Data]',
        '\U0001f680': '[Roadmap]',
        '\U0001f4cc': '[Overview]',
        '\u26a0': '[Alert]'
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode('ascii', 'ignore').decode('ascii')

class BeautifulResearchPDF(FPDF):
    """Clean, beautifully styled research report PDF."""
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, 'AMBER AI - Comprehensive Research Report', 0, 1, 'R')
        self.set_draw_color(220, 225, 230)
        self.set_line_width(0.3)
        self.line(18, 16, 192, 16)
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')

def generate_clean_pdf(report_text: str, output_path: str = "Final_Research_Report.pdf") -> str:
    """Renders structured markdown into an elegant, publication-grade PDF."""
    pdf = BeautifulResearchPDF()
    pdf.add_page()

    lines = report_text.splitlines()
    for raw_line in lines:
        safe_line = sanitize_latin1(raw_line).strip()

        if not safe_line:
            pdf.ln(3)
            continue

        # Top-level Title (# Title)
        if safe_line.startswith("# "):
            title_text = safe_line[2:].strip().replace("**", "")
            pdf.set_font('Helvetica', 'B', 16)
            pdf.set_text_color(20, 30, 55)
            pdf.multi_cell(0, 8, title_text)
            pdf.ln(3)
            continue

        # Section Header (## Header)
        if safe_line.startswith("## "):
            sec_text = safe_line[3:].strip().replace("**", "")
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(37, 99, 235)  # Royal Blue
            pdf.multi_cell(0, 7, sec_text)
            pdf.ln(1.5)
            continue

        # Sub-section (### Subheader)
        if safe_line.startswith("### "):
            sub_text = safe_line[4:].strip().replace("**", "")
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(0, 6, sub_text)
            pdf.ln(1)
            continue

        # Bullet items (* or -)
        if safe_line.startswith("* ") or safe_line.startswith("- "):
            bullet_text = safe_line[2:].strip().replace("**", "")
            pdf.set_font('Helvetica', '', 9.5)
            pdf.set_text_color(55, 65, 81)
            pdf.set_x(22)
            pdf.cell(4, 5, "-", 0, 0)
            pdf.multi_cell(0, 5, bullet_text)
            pdf.ln(1)
            continue

        # Normal paragraph text
        clean_para = safe_line.replace("**", "")
        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5.2, clean_para)
        pdf.ln(1.5)

    pdf.output(output_path)
    return output_path

def check_pdf_intent(user_prompt: str) -> tuple[bool, str]:
    """
    Detects if the user wants to generate/create a PDF.
    Returns: (is_pdf_intent, clean_topic_if_any)
    """
    prompt = user_prompt.strip()
    lower = prompt.lower()

    # 1. Pure PDF generation requests targeting prior conversation
    pure_patterns = [
        r"^(please\s+)?(generate|make|create|download|export|build|convert|save|give\s+me)\s+(the\s+|a\s+)?pdf(\s+report)?(\s+for\s+(this|the\s+report|it))?[\.\!\?]?$",
        r"^(pdf|generate\s+pdf|make\s+pdf|create\s+pdf|download\s+pdf|export\s+pdf|save\s+pdf|pdf\s+please)[\.\!\?]?$",
        r"^(can|could|would)\s+you\s+(please\s+)?(generate|make|create|download|export)\s+(a\s+|the\s+)?pdf(\s+for\s+this)?[\.\!\?]?$",
        r"^(i\s+want\s+a\s+pdf|convert\s+to\s+pdf|give\s+me\s+pdf)[\.\!\?]?$"
    ]
    for pat in pure_patterns:
        if re.search(pat, lower):
            return True, ""

    # 2. Check if prompt mentions PDF and either asks for PDF or pairs it with a research topic
    if "pdf" in lower:
        if any(phrase in lower for phrase in ["make it into a pdf", "save as pdf", "get this as pdf", "make pdf of this", "create pdf of this", "generate pdf of this"]):
            return True, ""

        # Hybrid: e.g. "research quantum computing and make pdf"
        cleaned = re.sub(r'\b(and\s+)?(please\s+)?(generate|make|create|download|export|save|convert)\s+(a\s+|the\s+)?pdf(\s+report)?\b', '', prompt, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\b(in|as)\s+pdf\b', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\bpdf\b', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'[\.\!\?]+$', '', cleaned).strip()
        if len(cleaned.split()) >= 2:
            return True, cleaned
        elif len(cleaned) == 0:
            return True, ""

    return False, prompt

@Research_AI.post("/chat")
def do_research(request: ResearchRequest, db: Session = Depends(get_session)):
        user_topic = request.user_topic.strip()
        print(f"\n⚡ API received request: '{user_topic}' (conversation_id={request.conversation_id}, project_id={request.project_id})...\n")

        # 1. Fetch chosen model dynamically:
        selected_llm = get_llm(request.model_id)
        report_agent.llm = selected_llm
        publisher_agent.llm = selected_llm

        # Use existing conversation if provided, or create a new one
        if request.conversation_id:
            conv = db.get(Conversation, request.conversation_id)
        else:
            conv = None

        if not conv:
            conv = Conversation(title=user_topic[:50])
            db.add(conv)
            db.commit()
            db.refresh(conv)

        # Link conversation to project if project_id is provided
        project_context = ""
        if request.project_id:
            project_obj = db.get(Project, request.project_id)
            if project_obj:
                if project_obj.instructions:
                    project_context = f" (Project Context/Instructions: {project_obj.instructions})"
                existing_link = db.exec(
                    select(ProjectConversation).where(
                        ProjectConversation.project_id == request.project_id,
                        ProjectConversation.conversation_id == conv.id
                    )
                ).first()
                if not existing_link:
                    db.add(ProjectConversation(project_id=request.project_id, conversation_id=conv.id))
                    db.commit()

        # Query prior messages for multi-turn conversational context
        prior_messages = db.exec(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.id.asc())
        ).all()

        history_context = ""
        if prior_messages:
            history_lines = []
            for m in prior_messages[-6:]:
                role = "User" if m.sender == "user" else "Assistant"
                snippet = m.text.strip()
                if len(snippet) > 800:
                    snippet = snippet[:800] + "... [prior report summary]"
                history_lines.append(f"{role}: {snippet}")
            if history_lines:
                history_context = "\n\n### 💬 PRIOR CONVERSATION HISTORY & MULTI-TURN CONTEXT:\n" + "\n".join(history_lines) + "\n\n(Note: Incorporate context from the prior conversation history above to answer follow-up questions, maintain consistency, and resolve references to prior topics.)\n"

        # Save the user's message to this conversation thread
        user_msg = Message(conversation_id=conv.id, sender="user", text=user_topic)
        db.add(user_msg)
        db.commit()

        # Check if user prompt is asking to generate a PDF
        is_pdf_cmd, clean_topic = check_pdf_intent(user_topic)

        # Case A: Pure PDF command (e.g. "make pdf", "generate pdf", "download pdf")
        if is_pdf_cmd and not clean_topic:
            print(f"📄 [PDF REQUEST DETECTED] Generating PDF for conversation {conv.id}...")
            reports = db.exec(
                select(Message)
                .where(Message.conversation_id == conv.id, Message.sender == "ai")
                .order_by(Message.id.desc())
            ).all()

            target_report = None
            for m in reports:
                if m.text and not m.text.startswith("Error:") and not m.text.startswith("Research was terminated") and "PDF Generated Successfully" not in m.text and "No Prior Research Report" not in m.text:
                    target_report = m.text
                    break

            if target_report:
                pdf_path = "Final_Research_Report.pdf"
                generate_clean_pdf(target_report, pdf_path)

                pdf_reply = (
                    "# 📄 Research Report PDF Generated Successfully!\n\n"
                    "Your comprehensive research report has been compiled and saved as **Final_Research_Report.pdf**.\n\n"
                    "### 📥 Direct Download\n"
                    "👉 **[Click Here to Download Final_Research_Report.pdf](/api/research/download-pdf)**\n\n"
                    "> 💡 *The PDF includes the full executive summary, technical architecture, empirical benchmarks, base paper references, and related research directions formatted with elegant typography and page numbers.*"
                )
                ai_msg = Message(conversation_id=conv.id, sender="ai", text=pdf_reply)
                db.add(ai_msg)
                db.commit()

                return {
                    "status": "Success",
                    "message": "PDF generated successfully!",
                    "topic": user_topic,
                    "report": pdf_reply,
                    "conversation_id": conv.id,
                    "is_pdf_ready": True,
                    "pdf_url": "/api/research/download-pdf",
                    "needs_approval": False
                }
            else:
                no_report_reply = (
                    "# ℹ️ No Prior Research Report Found\n\n"
                    "I haven't generated a research report in this conversation yet to convert into a PDF.\n\n"
                    "Please enter a topic to research first (e.g. *'Research Quantum Computing in Cryptography'* or *'Solid-State Battery Innovations'*), and once the report is ready, simply type **'generate pdf'** or **'make pdf'**!"
                )
                ai_msg = Message(conversation_id=conv.id, sender="ai", text=no_report_reply)
                db.add(ai_msg)
                db.commit()

                return {
                    "status": "Info",
                    "message": "No report found to convert to PDF.",
                    "topic": user_topic,
                    "report": no_report_reply,
                    "conversation_id": conv.id,
                    "is_pdf_ready": False,
                    "needs_approval": False
                }

        # Case B: Hybrid or Standard Research Request
        research_topic = clean_topic if (is_pdf_cmd and clean_topic) else user_topic

        # 2. Fast Direct Python Search (<0.8s) - Perplexity Architecture
        print(f"🔍 [FAST PYTHON SEARCH] Directly querying live web & academic sources for: {research_topic}...")
        search_context = execute_python_search(research_topic)

        # 3. Comprehensive, High-Detail Long-Form Synthesis Task Description
        task_description = (
            f"You are the Chief Technical Research Scientist and Lead Executive Editor.\n"
            f"Synthesize the verified real-world research sources into an exhaustive, highly detailed, textbook-quality research document on: {research_topic}{project_context}.\n"
            f"{history_context}\n"
            f"### 🌐 VERIFIED SEARCH SOURCES & BASE PAPERS:\n{search_context}\n\n"
            "CRITICAL INSTRUCTIONS FOR DEPTH & COMPREHENSIVENESS:\n"
            "1. EXHAUSTIVE LONG-FORM CONTENT: Write a rich, expansive, in-depth report (aim for 1,200 to 2,000+ words). Do not cut corners, do not use brief summaries, and do not provide superficial answers. Explain concepts thoroughly with educational depth so the reader gains complete, nuanced understanding.\n"
            "2. CRYSTAL-CLEAR ACCESSIBILITY: Balance advanced technical rigor with accessible explanations, intuitive analogies, and clean formatting so that any user can easily read and understand.\n"
            "3. STRUCTURED SUBHEADINGS & DATA: Use bold markdown, bullet points, subheadings (###), and comparison tables wherever appropriate.\n\n"
            "STRICT 7-SECTION STRUCTURE REQUIRED:\n\n"
            "# [Authoritative & Insightful Title]\n\n"
            "## 📌 1. Comprehensive Executive Overview & Foundational Concepts\n"
            "(Provide a thorough multi-paragraph introduction to the topic. Explain historical context, the core problem or scientific breakthrough that led to its discovery, intuitive real-world analogies to make the core ideas easy to understand, and its transformative significance in the current technological or scientific landscape.)\n\n"
            "## 🔬 2. Deep-Dive Technical Architecture & Mechanism Breakdown\n"
            "(Provide an exhaustive breakdown of the underlying technology, system, or scientific mechanisms. Divide into granular subheadings (###) for each core sub-system, component, or operational stage. Explain step-by-step how inputs transform into outputs, key operational algorithms, specifications, and underlying mathematical or physical principles.)\n\n"
            "## 📊 3. Empirical Evidence, Performance Benchmarks & Real-World Implementations\n"
            "(Present rich empirical data, quantitative metrics, and performance benchmarks comparing this solution against conventional baselines. Include markdown comparison tables and bulleted statistics. Detail real-world case studies, pilot deployments, or landmark laboratory results.)\n\n"
            "## ⚠️ 4. Critical Engineering Bottlenecks, Limitations & Roadblocks\n"
            "(Objectively analyze current engineering challenges, scalability hurdles, cost factors, thermal/stability issues, safety constraints, or regulatory bottlenecks. Contrast current laboratory prototype status with what is required for full-scale commercialization.)\n\n"
            "## 🚀 5. Strategic Future Roadmap & Emerging Horizons (Next 3–5 Years)\n"
            "(Detail upcoming innovations, next-generation paradigms, ongoing experimental breakthroughs, and key inflection points to watch over the next 3 to 5 years.)\n\n"
            "## 📚 6. Base Papers & Authoritative Reference Sources\n"
            "(List 4 to 6 real base papers or authoritative sources extracted directly from the verified search results above. Format each strictly as:)\n"
            "* [Exact Paper Title or Source Organization](Direct URL) - In-depth breakdown explaining what this specific study discovered, its methodology, and why it is authoritative.\n\n"
            "## 💡 7. High-Impact Related Research Topics & Exploration Directions\n"
            "(List 4 to 5 specific, compelling follow-up research questions or sub-topics as bullet points.)\n"
            "* **Topic 1:** Brief exploration note\n"
            "* **Topic 2:** Brief exploration note\n"
            "* **Topic 3:** Brief exploration note\n"
            "* **Topic 4:** Brief exploration note\n"
        )

        task_id = request.task_id or str(uuid.uuid4())
        print(f"\n⚡ Launching CrewAI process for: {research_topic} [Task ID: {task_id}]...\n")

        res_queue = mp.Queue()
        proc = mp.Process(
            target=_run_crew_synthesis,
            args=(task_description, request.model_id or "nvidia_llama", res_queue),
            daemon=True
        )
        active_tasks[task_id] = proc
        proc.start()

        while proc.is_alive():
            proc.join(timeout=0.2)

        active_tasks.pop(task_id, None)

        try:
            status, result = res_queue.get_nowait()
        except Exception:
            status, result = "aborted", "Research was terminated by user."

        if status == "aborted":
            return {
                "status": "Terminated",
                "message": "Research was terminated by user.",
                "topic": user_topic,
                "report": "Research was terminated by user.",
                "conversation_id": conv.id,
                "needs_approval": False,
                "is_pdf_ready": False
            }
        elif status == "error":
            return {
                "status": "Error",
                "message": f"Execution error: {result}",
                "topic": user_topic,
                "report": f"Error: {result}",
                "conversation_id": conv.id,
                "needs_approval": False,
                "is_pdf_ready": False
            }

        result_text = str(result)

        # If user explicitly asked for PDF in the prompt (hybrid mode), compile PDF immediately
        if is_pdf_cmd:
            generate_clean_pdf(result_text, "Final_Research_Report.pdf")
            result_text += "\n\n---\n### 📄 Research Report PDF Created\n👉 **[Click Here to Download Final_Research_Report.pdf](/api/research/download-pdf)**\n"

        ai_msg = Message(conversation_id=conv.id, sender="ai", text=result_text)
        db.add(ai_msg)
        db.commit()

        return {
            "status": "Success",
            "message": "Report generated successfully!",
            "topic": user_topic,
            "report": result_text,
            "conversation_id": conv.id,
            "needs_approval": False,
            "is_pdf_ready": is_pdf_cmd,
            "pdf_url": "/api/research/download-pdf" if is_pdf_cmd else None
        }

@Research_AI.post("/approve-pdf")
def approve_and_generate_pdf(req: ApproveReportRequest, db: Session = Depends(get_session)):
    """Generates styled Final_Research_Report.pdf from provided report content."""
    report_content = req.report_text
    print("\n📄 Generating Final_Research_Report.pdf...\n")

    try:
        pdf_path = "Final_Research_Report.pdf"
        generate_clean_pdf(report_content, pdf_path)

        if req.conversation_id:
            confirm_msg = Message(
                conversation_id=req.conversation_id,
                sender="ai",
                text="✅ Final_Research_Report.pdf has been generated successfully and is ready for download."
            )
            db.add(confirm_msg)
            db.commit()

        return {
            "status": "Success",
            "message": "Final_Research_Report.pdf generated successfully!",
            "pdf_url": "/api/research/download-pdf",
            "filename": "Final_Research_Report.pdf"
        }
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return {"status": "Error", "message": f"Failed to generate PDF: {str(e)}"}

@Research_AI.get("/download-pdf")
def download_final_pdf():
    """Allows user to download the approved Final_Research_Report.pdf."""
    pdf_path = "Final_Research_Report.pdf"
    if not os.path.exists(pdf_path):
        return {"status": "Error", "message": "PDF not generated yet"}
    return FileResponse(pdf_path, media_type="application/pdf", filename="Final_Research_Report.pdf")

@Research_AI.post("/terminate")
def terminate_research_task(req: TerminateRequest):
    """Immediately kill the running background CrewAI process or streaming task on user cancellation."""
    task_id = req.task_id
    print(f"⚡ Received request to terminate backend task: {task_id}")
    active_streaming_tasks[task_id] = False
    proc = active_tasks.pop(task_id, None)
    if proc:
        kill_process(proc)
    return {"status": "Success", "message": f"Task {task_id} terminated."}

@Research_AI.get("/conversations")
def get_all_conversations(db: Session = Depends(get_session)):
    statement = select(Conversation).order_by(Conversation.id.desc())
    conversations = db.exec(statement).all()
    return conversations

# 2. Fetch all messages for a specific conversation
@Research_AI.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: int, db: Session = Depends(get_session)):
    statement = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id.asc())
    messages = db.exec(statement).all()
    return messages

# 3. Projects Endpoints (ChatGPT Projects Workflow)
@Research_AI.get("/projects")
def get_all_projects(db: Session = Depends(get_session)):
    SQLModel.metadata.create_all(engine)
    projects = db.exec(select(Project).order_by(Project.id.asc())).all()
    if not projects:
        default_proj = Project(name="CLI Project", instructions="Focus on command-line tools, clean architecture, and developer workflows.")
        db.add(default_proj)
        db.commit()
        db.refresh(default_proj)
        projects = [default_proj]

    links = db.exec(select(ProjectConversation)).all()
    proj_map: dict[int, list[int]] = {}
    for link in links:
        proj_map.setdefault(link.project_id, []).append(link.conversation_id)

    return [
        {
            "id": p.id,
            "name": p.name,
            "instructions": p.instructions or "",
            "conversation_ids": proj_map.get(p.id, [])
        }
        for p in projects
    ]

@Research_AI.post("/projects")
def create_project(req: ProjectCreateRequest, db: Session = Depends(get_session)):
    SQLModel.metadata.create_all(engine)
    new_proj = Project(name=req.name.strip(), instructions=(req.instructions or "").strip())
    db.add(new_proj)
    db.commit()
    db.refresh(new_proj)
    return {
        "id": new_proj.id,
        "name": new_proj.name,
        "instructions": new_proj.instructions or "",
        "conversation_ids": []
    }

@Research_AI.put("/projects/{project_id}")
def update_project(project_id: int, req: ProjectCreateRequest, db: Session = Depends(get_session)):
    proj = db.get(Project, project_id)
    if not proj:
        return {"status": "Error", "message": "Project not found"}
    proj.name = req.name.strip()
    proj.instructions = (req.instructions or "").strip()
    db.add(proj)
    db.commit()
    db.refresh(proj)
    links = db.exec(select(ProjectConversation).where(ProjectConversation.project_id == project_id)).all()
    return {
        "id": proj.id,
        "name": proj.name,
        "instructions": proj.instructions or "",
        "conversation_ids": [l.conversation_id for l in links]
    }

@Research_AI.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_session)):
    proj = db.get(Project, project_id)
    if proj:
        links = db.exec(select(ProjectConversation).where(ProjectConversation.project_id == project_id)).all()
        for link in links:
            db.delete(link)
        db.delete(proj)
        db.commit()
    return {"status": "Success"}

@Research_AI.post("/projects/{project_id}/conversations")
def assign_conversation_to_project(project_id: int, req: ProjectAssignConversationRequest, db: Session = Depends(get_session)):
    existing = db.exec(
        select(ProjectConversation).where(
            ProjectConversation.project_id == project_id,
            ProjectConversation.conversation_id == req.conversation_id
        )
    ).first()
    if not existing:
        db.add(ProjectConversation(project_id=project_id, conversation_id=req.conversation_id))
        db.commit()
    return {"status": "Success"}