# server/app/services/chat_service.py
from sqlalchemy.orm import Session
from ..db import models as db_models
from ..openai_models import ChatMessageInput # For type hinting message history
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def get_or_create_user(db: Session, user_identifier: str) -> db_models.User:
    user = db.query(db_models.User).filter(db_models.User.id == user_identifier).first()
    if not user:
        user = db_models.User(id=user_identifier, display_name=user_identifier)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"ChatService: User '{user_identifier}' created.")
    return user

def get_or_create_project(db: Session, project_name: str, description: Optional[str] = None) -> db_models.Project:
    project = db.query(db_models.Project).filter(db_models.Project.name == project_name).first()
    if not project:
        project = db_models.Project(name=project_name, description=description)
        db.add(project)
        db.commit()
        db.refresh(project)
        logger.info(f"ChatService: Project '{project_name}' (ID: {project.id}) created.")
    return project

def ensure_user_linked_to_project(db: Session, user_id: str, project_id: str):
    link = db.query(db_models.ProjectUserLink).filter_by(user_id=user_id, project_id=project_id).first()
    if not link:
        link = db_models.ProjectUserLink(user_id=user_id, project_id=project_id)
        db.add(link)
        db.commit()
        logger.info(f"ChatService: User '{user_id}' linked to project '{project_id}'.")

def get_or_create_conversation(
    db: Session,
    conversation_id_from_owi: str,
    project_id: str,
    creator_user_id: str,
    title: Optional[str] = "New Chat"
) -> db_models.Conversation:
    conversation = db.query(db_models.Conversation).filter(
        db_models.Conversation.id == conversation_id_from_owi,
        db_models.Conversation.project_id == project_id
    ).first()

    if not conversation:
        # User and Project should already exist if this point is reached via normal flow
        conversation = db_models.Conversation(
            id=conversation_id_from_owi,
            title=title if title else "New Chat",
            project_id=project_id,
            creator_user_id=creator_user_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        logger.info(f"ChatService: Conversation '{conversation_id_from_owi}' created in project '{project_id}'.")
    return conversation

def add_message_to_database(
    db: Session,
    conversation_id: str,
    role: str, 
    content: str,
    author_user_id: Optional[str] = None, 
    model_used: Optional[str] = None
) -> db_models.Message:
    if role == "user" and not author_user_id:
        logger.warning(f"Attempting to save user message for conversation '{conversation_id}' without author_user_id.")
        
    db_message = db_models.Message(
        conversation_id=conversation_id,
        author_user_id=author_user_id if role == "user" else None,
        role=role,
        content=content,
        model_used=model_used if role == "assistant" else None
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    logger.debug(f"ChatService: Message (Role: {role}) by '{author_user_id if author_user_id else 'assistant'}' added to conv '{conversation_id}'.")
    return db_message

def get_message_history_for_llm_context(db: Session, conversation_id: str, last_n_messages: int = 10) -> List[ChatMessageInput]:
    db_hist_messages = db.query(db_models.Message)\
                         .filter(db_models.Message.conversation_id == conversation_id)\
                         .order_by(db_models.Message.timestamp.desc())\
                         .limit(last_n_messages)\
                         .all()
    
    llm_history: List[ChatMessageInput] = []
    for msg in reversed(db_hist_messages): 
        llm_history.append(ChatMessageInput(role=msg.role, content=msg.content))
    logger.debug(f"Retrieved {len(llm_history)} messages for LLM context from conversation '{conversation_id}'.")
    return llm_history
