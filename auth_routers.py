from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, Field, select
from database import engine
from model import User

auth_route = APIRouter()

@auth_route.post("/register")
def register_user(username: str):
    with Session(engine) as session:
        existing_user = session.exec(select(User).where(User.username == username)).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

        new_user = User(username=username)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        
        return {"message": "User registered successfully", "user_id": new_user.id}

@auth_route.post("/login")
def login_user(username: str):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "Login successful", "user_id": user.id}