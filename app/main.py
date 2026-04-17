from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
from sqlalchemy import asc

from .db import get_db, engine
from .models import Base, Conversation, Message
from .services import get_rep_explanation, get_rep_data
from rag.pipeline import RAGSystem

# ---------------- INIT APP ----------------

app = FastAPI()

# Create DB tables
Base.metadata.create_all(bind=engine)

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT RAG ----------------

rag = RAGSystem()
rag.build()

# ---------------- REQUEST MODELS ----------------

class ChatRequest(BaseModel):
    question: str
    conversation_id: str

class ConversationRequest(BaseModel):
    rep_id: str

# ---------------- ROOT ----------------

@app.get("/")
def root():
    return {"message": "IC Chatbot Backend Running"}

# ---------------- START CONVERSATION ----------------

@app.post("/conversation/start")
def start_conversation(data: ConversationRequest, db: Session = Depends(get_db)):
    conversation_id = str(uuid4())

    conv = Conversation(
        conversation_id=conversation_id,
        rep_id=data.rep_id
    )

    db.add(conv)
    db.commit()

    return {"conversation_id": conversation_id}

# ---------------- CHAT ----------------

@app.post("/chat/{rep_id}")
def chat(rep_id: str, data: ChatRequest, db: Session = Depends(get_db)):

    question = data.question
    conversation_id = data.conversation_id

    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        message_text=question
    )
    db.add(user_msg)
    db.commit()

    # Generate response
    answer = get_rep_explanation(rep_id, question, db, rag)

    if answer == "Rep not found.":
        raise HTTPException(status_code=404, detail="Rep not found")

    clean_answer = " ".join(answer.replace("\n", " ").split())

    # Save bot response
    bot_msg = Message(
        conversation_id=conversation_id,
        role="bot",
        message_text=clean_answer
    )
    db.add(bot_msg)
    db.commit()

    return {"answer": clean_answer}

# ---------------- REP DATA ----------------

@app.get("/rep/{rep_id}")
def rep_data(rep_id: str, db: Session = Depends(get_db)):
    rep = get_rep_data(rep_id, db)

    if not rep:
        raise HTTPException(status_code=404, detail="Rep not found")

    return rep