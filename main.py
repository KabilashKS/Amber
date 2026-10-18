from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from server import Research_AI
from auth_routers import auth_route
from sqlmodel import Field, SQLModel
from database import engine
import model

app = FastAPI(title = "Research AI")
templates = Jinja2Templates(directory="templates")
app.include_router(Research_AI)
app.include_router(auth_route, prefix="/auth", tags=["Authentication"])


SQLModel.metadata.create_all(engine)

LOGO_FILE_PATH = "/home/kabilash/.gemini/antigravity/brain/83399c15-8e53-4883-bc19-86fc1b252c71/amber_logo_transparent.png"

@app.get("/api/research/logo")
@app.get("/home/kabilash/.gemini/antigravity/brain/83399c15-8e53-4883-bc19-86fc1b252c71/amber_logo_transparent.png")
async def serve_amber_logo():
    return FileResponse(LOGO_FILE_PATH, media_type="image/png")

@app.get("/")
async def home(request: Request):
    # This serves your UI when you open http://localhost:8000
    return templates.TemplateResponse(request=request, name="ui.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)