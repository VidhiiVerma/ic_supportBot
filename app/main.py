import json
import os
import asyncio
import logging
from typing import Optional
from datetime import datetime
from uuid import uuid4

# FastAPI imports
from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Database imports
from sqlalchemy.orm import Session
from sqlalchemy import asc

# Bot Framework imports
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)

from botbuilder.schema import Activity, ActivityTypes

# Your existing imports
from .db import get_db, engine
from .models import Base, Conversation, Message
from .services import get_rep_explanation, get_rep_data
from rag.pipeline import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_APP_ID = os.getenv("MICROSOFT_APP_ID", "YOUR_BOT_APP_ID")
BOT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "YOUR_BOT_APP_PASSWORD")
ALLOWED_TEAMS_IDS = os.getenv("ALLOWED_TEAMS_IDS", "").split(",")  # Optional: restrict to specific teams

# FASTAPI init
app = FastAPI(
    title="IC Support Bot",
    version="1.0.0",
    description="Teams Bot for IC Support"
)

# Create database
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://teams.microsoft.com",
        "https://teams.microsoft.us",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# initalize RAG
rag = RAGSystem()
rag.build()
logger.info("RAG System initialized")

# initalize BOT 
try:
    SETTINGS = BotFrameworkAdapterSettings(
        app_id=BOT_APP_ID,
        app_password=BOT_APP_PASSWORD
    )
    ADAPTER = BotFrameworkAdapter(SETTINGS)
    logger.info("Bot Framework Adapter initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Bot Framework: {str(e)}")
    ADAPTER = None


class AskRequest(BaseModel):
    """API endpoint for direct calls (not Teams)"""
    query: str = Field(..., min_length=1, max_length=500)
    rep_id: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None

    class Config:
        example = {
            "query": "What is my commission?",
            "rep_id": "rep_12345"
        }

class ChatRequest(BaseModel):
    """Continue an existing conversation"""
    question: str = Field(..., min_length=1)
    conversation_id: str

class ConversationRequest(BaseModel):
    """Start a new conversation"""
    rep_id: str

class AskResponse(BaseModel):
    """Response from /ask endpoint"""
    text: str
    conversation_id: str
    status: str = "success"

async def handle_teams_message(turn_context: TurnContext) -> None:
    """
    Main handler for Teams messages
    
    What happens here:
    1. Extract user's message from Teams
    2. Extract rep_id (the user who is using the bot)
    3. Call our /ask logic
    4. Send response back to Teams
    """
    
    try:
        # 1: Extract message info from Teams
        user_message = turn_context.activity.text  # What user type
        user_id = turn_context.activity.from_property.id  # Teams user ID
        user_name = turn_context.activity.from_property.name  # User's name
        channel_id = turn_context.activity.channel_id  # Should be "msteams"
        
        logger.info(f"Message from {user_name} ({user_id}): {user_message}")
        
        # STEP 2: Get rep_id from Teams metadata 
        # You have 3 ways to get rep_id:
        # Option A: From channel_data (set when installing bot)
        channel_data = turn_context.activity.channel_data or {}
        rep_id = channel_data.get("rep_id")
        
        # Option B: From user ID mapping (if you have a mapping table)
        # rep_id = await get_rep_id_from_teams_user(user_id)
        
        # Option C: Extract from Teams conversation (simplified)
        if not rep_id:
            # Fallback: use first 20 chars of user ID as temp rep_id
            rep_id = user_id[:20]
            logger.warning(f"No rep_id found, using Teams user ID: {rep_id}")
        
        # ===== STEP 3: Get conversation ID from Teams =====
        # Teams gives us a conversation reference we can use
        conversation_id = turn_context.activity.conversation.id
        
        # ===== STEP 4: Save to database =====
        db = next(get_db())  # Get DB session
        
        try:
            # Check if conversation exists in our DB
            conv = db.query(Conversation).filter_by(
                conversation_id=conversation_id
            ).first()
            
            if not conv:
                # Create new conversation record
                conv = Conversation(
                    conversation_id=conversation_id,
                    rep_id=rep_id
                )
                db.add(conv)
                db.commit()
            
            # Save user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                message_text=user_message
            )
            db.add(user_msg)
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database error: {str(e)}")
            await turn_context.send_activity("Error saving message. Please try again.")
            return
        
        # ===== STEP 5: Generate response using RAG =====
        try:
            answer = get_rep_explanation(rep_id, user_message, db, rag)
            
            # Clean answer
            if answer == "Rep not found.":
                clean_answer = "Sorry, I couldn't find information about this rep. Please check the rep ID."
            else:
                clean_answer = " ".join(answer.replace("\n", " ").split())[:2000]
            
        except Exception as e:
            logger.error(f"RAG error: {str(e)}")
            clean_answer = "Sorry, I encountered an error processing your request. Please try again."
        
        # ===== STEP 6: Save bot response =====
        try:
            bot_msg = Message(
                conversation_id=conversation_id,
                role="bot",
                message_text=clean_answer
            )
            db.add(bot_msg)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save bot message: {str(e)}")
        
        # ===== STEP 7: Send response back to Teams =====
        await turn_context.send_activity(clean_answer)
        logger.info(f"Sent response to {user_name}")
        
    except Exception as e:
        logger.error(f"Unhandled error in message handler: {str(e)}", exc_info=True)
        await turn_context.send_activity("An error occurred. Please try again.")

# ==================== ERROR HANDLER ====================

async def on_error(context: TurnContext, error: Exception):
    """Called when an error occurs in turn processing"""
    logger.error(f"Error in turn: {str(error)}", exc_info=True)
    await context.send_activity(f"A bot error occurred: {str(error)}")

# Set error handler
if ADAPTER:
    ADAPTER.on_turn_error = on_error

# ==================== ENDPOINTS ====================

# -------- HEALTH CHECK --------
@app.get("/health")
def health_check():
    """Check if bot is running"""
    return {
        "status": "healthy",
        "service": "ic-support-bot",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# -------- ROOT --------
@app.get("/")
def root():
    return {
        "message": "IC Support Bot Running",
        "framework": "Teams Bot Framework",
        "endpoints": {
            "health": "/health",
            "teams_messages": "/api/messages",
            "direct_ask": "/ask"
        }
    }

# -------- TEAMS BOT MESSAGES ENDPOINT (THE MAIN ONE) --------
@app.post("/api/messages")
async def messages(req: Request):
    if not ADAPTER:
        logger.error("Bot Framework not initialized")
        return Response(status_code=500)

    try:
        # Parse incoming request
        body = await req.json()
        logger.info(f"Received activity: {body.get('type')}")

        auth_header = req.headers.get("Authorization", "")

        # ✅ FIX: correct way to call process_activity
        activity = Activity().deserialize(body)

        response = await ADAPTER.process_activity(
            activity,
            auth_header,
            handle_teams_message
        )

        # Return response to Teams
        if response:
            return response
        else:
            return Response(status_code=201)

    except Exception as e:
        logger.error(f"Error in /api/messages: {str(e)}", exc_info=True)
        return Response(
            status_code=500,
            content=json.dumps({"error": str(e)})
        )
# -------- DIRECT /ASK ENDPOINT (for external integrations) --------
@app.post("/ask", response_model=AskResponse)
def ask(
    data: AskRequest,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    """
    Simplified endpoint for direct API calls (not Teams)
    
    Use this if you want to call the bot from another system
    You can also call this from Copilot Studio if needed
    """
    
    # Optional: Verify API key
    # expected_key = os.getenv("API_KEY")
    # if authorization != f"Bearer {expected_key}":
    #     raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Validate rep exists
    rep = get_rep_data(data.rep_id, db)
    if not rep:
        raise HTTPException(
            status_code=404,
            detail=f"Rep {data.rep_id} not found"
        )
    
    # Use provided conversation or create new
    conversation_id = data.conversation_id or str(uuid4())
    
    try:
        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            message_text=data.query
        )
        db.add(user_msg)
        
        # Generate answer
        answer = get_rep_explanation(data.rep_id, data.query, db, rag)
        clean_answer = " ".join(answer.replace("\n", " ").split())
        
        # Save bot response
        bot_msg = Message(
            conversation_id=conversation_id,
            role="bot",
            message_text=clean_answer
        )
        db.add(bot_msg)
        db.commit()
        
        return AskResponse(
            text=clean_answer,
            conversation_id=conversation_id,
            status="success"
        )
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------- START CONVERSATION --------
@app.post("/conversation/start")
def start_conversation(
    data: ConversationRequest,
    db: Session = Depends(get_db)
):
    """Start a new tracked conversation"""
    conversation_id = str(uuid4())
    
    conv = Conversation(
        conversation_id=conversation_id,
        rep_id=data.rep_id
    )
    
    db.add(conv)
    db.commit()
    
    return {"conversation_id": conversation_id}

# -------- CHAT (Continue conversation) --------
@app.post("/chat/{rep_id}")
def chat(
    rep_id: str,
    data: ChatRequest,
    db: Session = Depends(get_db)
):
    """Continue an existing conversation (not usually used with Teams bot)"""
    
    question = data.question
    conversation_id = data.conversation_id
    
    # Validate conversation exists
    conv = db.query(Conversation).filter_by(
        conversation_id=conversation_id,
        rep_id=rep_id
    ).first()
    
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    try:
        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            message_text=question
        )
        db.add(user_msg)
        
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
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error in /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------- REP DATA --------
@app.get("/rep/{rep_id}")
def rep_data(rep_id: str, db: Session = Depends(get_db)):
    """Get rep-specific data"""
    rep = get_rep_data(rep_id, db)
    
    if not rep:
        raise HTTPException(status_code=404, detail="Rep not found")
    
    return rep

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Run when app starts"""
    logger.info("🚀 IC Support Bot starting up")
    logger.info(f"Bot App ID: {BOT_APP_ID}")
    logger.info("Bot Framework adapter ready to receive messages from Teams")

@app.on_event("shutdown")
async def shutdown_event():
    """Run when app shuts down"""
    logger.info("🛑 IC Support Bot shutting down")

# ==================== FOR LOCAL TESTING ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3978,
        reload=False,
        log_level="info"
    )
