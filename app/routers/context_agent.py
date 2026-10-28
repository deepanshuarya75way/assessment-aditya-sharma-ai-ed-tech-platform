from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from OAuth2 import OAuth2
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from core.schemas import ChatRequest
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from core.chat_history_db import chat_sessions, messages
from app.agents.context_agent.graph import agent
from ..retriever.hybrid_retrieval import ensemble_retriever, register_retriever
from ..retriever.process_docs import process_document
import asyncio

load_dotenv()


UPLOAD_DIR = Path(__file__).resolve().parents[1] / "load_documents" / "uploads"

router = APIRouter(
    prefix="/rag-agent",
    tags=["/"],
)

@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    current_user: int = Depends(OAuth2.get_current_user)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIR / f"{uuid4()}{Path(file.filename).suffix}"

    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)


        
        chunks = await asyncio.to_thread(
            process_document, str(file_path), current_user,
            file.filename
        )

        retriever = ensemble_retriever(chunks)

        register_retriever(current_user, retriever)
        return {
            'message': 'Documents uploaded'
            }
    except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )

    except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )

    finally:
        if file_path.exists():
            file_path.unlink()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: int = Depends(OAuth2.get_current_user)
):
    

    if request.session_id is None:

        session_id = str(uuid4())

        chat_sessions.insert_one({
            "_id": session_id,
            "user_id": str(current_user),
            "title": request.user_query[:30],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

    else:
        session_id = request.session_id


    messages.insert_one({
        "_id": str(uuid4()),
        "session_id": session_id,
        "role": "user",
        "content": request.user_query,
        "created_at": datetime.now()
    })


    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    ini = {
        "id": current_user,
        "user_query": request.user_query,
        "use_strictly_retriever": request.use_strictly_retriever,
        "docs_available": request.docs_available,
    }

    result = await run_in_threadpool(
        agent.invoke,
        ini,
        config=config
    )

    assistant_response = (
        result.get("final_answer")
        or (result.get("chat_history")[-1].content if result.get("chat_history") else None)
        or result.get("rewrite_query")
        or "I apologize, but no answer could be compiled for your request."
    )
    result["final_answer"] = assistant_response


    messages.insert_one({
        "_id": str(uuid4()),
        "session_id": session_id,
        "role": "assistant",
        "content": assistant_response,
        "created_at": datetime.utcnow()
    })

    # Update chat activity timestamp
    chat_sessions.update_one(
        {"_id": session_id},
        {
            "$set": {
                "updated_at": datetime.utcnow()
            }
        }
    )

    return {
        "session_id": session_id,
        "result": result
    }


@router.get("/sessions")
async def get_sessions(
    current_user: int = Depends(OAuth2.get_current_user)
):
    sessions = list(
        chat_sessions.find(
            {"user_id": str(current_user)}
        ).sort("updated_at", -1)
    )

    return sessions


@router.get("/messages/{session_id}")
async def get_messages(
    session_id: str,
    current_user: int = Depends(OAuth2.get_current_user)
):
    session = chat_sessions.find_one({
        "_id": session_id,
        "user_id": str(current_user)
    })

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    history = list(
        messages.find(
            {"session_id": session_id}
        ).sort("created_at", 1)
    )

    return history