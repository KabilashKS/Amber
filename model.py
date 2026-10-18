from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "users"  
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True)

class ChatLog(SQLModel, table=True):
    __tablename__ = "chat_logs"
    id: int | None = Field(default=None, primary_key=True)
    user_id: int
    topic: str
    ai_report: str

class Conversation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    
class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int
    sender: str  # "user" or "ai"
    text: str

class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    instructions: str | None = Field(default="")

class ProjectConversation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int
    conversation_id: int