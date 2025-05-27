# server/app/services/chat_service.py
from sqlalchemy.orm import Session
from ..db import models as db_models
from ..openai_models import ChatMessageInput # For type hinting message history
from typing import List, Optional

def get_or_create_user(db: Session, user_identifier: str) -> db_models.User:
    """Gets a user by their OWI identifier, or creates them if they don't exist."""
    user = db.query(db_models.User).filter(db_models.User.id == user_identifier).first()
    if not user:
        user = db_models.User(id=user_identifier, display_name=user_identifier) # Simple mapping for now
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"ChatService: Created user '{user_identifier}'")
    return user

def get_or_create_project(db: Session, project_name: str, description: Optional[str] = None) -> db_models.Project:
    """Gets a project by name, or creates it if it doesn't exist."""
    project = db.query(db_models.Project).filter(db_models.Project.name == project_name).first()
    if not project:
        project = db_models.Project(name=project_name, description=description)
        db.add(project)
        db.commit()
        db.refresh(project)
        print(f"ChatService: Created project '{project_name}' with ID {project.id}")
    return project

def ensure_user_in_project(db: Session, user_id: str, project_id: str):
    """Ensures a user is linked to a project."""
    link = db.query(db_models.ProjectUserLink).filter_by(user_id=user_id, project_id=project_id).first()
    if not link:
        link = db_models.ProjectUserLink(user_id=user_id, project_id=project_id)
        db.add(link)
        db.commit()
        print(f"ChatService: Linked user '{user_id}' to project '{project_id}'")

def get_or_create_conversation(
    db: Session,
    conversation_id_from_owi: str,
    project_id: str,
    creator_user_id: str, # This is the user_identifier from OWI
    title: Optional[str] = "New Chat"
) -> db_models.Conversation:
    """Gets a conversation by its OWI ID, or creates it if it doesn't exist for the given project."""
    # Ensure user and project entities exist first
    get_or_create_user(db, creator_user_id)
    # project = get_or_create_project(db, project_name) # project_id is already assumed to be valid

    conversation = db.query(db_models.Conversation).filter(
        db_models.Conversation.id == conversation_id_from_owi,
        db_models.Conversation.project_id == project_id
    ).first()

    if not conversation:
        conversation = db_models.Conversation(
            id=conversation_id_from_owi,
            title=title,
            project_id=project_id,
            creator_user_id=creator_user_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"ChatService: Created conversation '{conversation_id_from_owi}' in project '{project_id}'")
    return conversation

def add_message_to_db(
    db: Session,
    conversation_id: str, # This is Conversation.id (OWI's chat_id)
    role: str, # "user" or "assistant"
    content: str,
    author_user_id: Optional[str] = None, # OWI username for 'user' role
    model_used: Optional[str] = None
) -> db_models.Message:
    """Adds a message to the specified conversation in the database."""
    # If it's a user message, ensure the user exists (should have been handled by get_or_create_conversation for creator)
    if role == "user" and author_user_id:
        get_or_create_user(db, author_user_id) # Ensure author exists if it's a user
    
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
    # print(f"ChatService: Added '{role}' message to conversation '{conversation_id}'")
    return db_message

def get_message_history_for_llm(db: Session, conversation_id: str, last_n: int = 10) -> List[ChatMessageInput]:
    """
    Retrieves the last N messages for a conversation, formatted for LLM context.
    Orders by timestamp ascending (oldest to newest).
    """
    db_messages = db.query(db_models.Message)\
                    .filter(db_models.Message.conversation_id == conversation_id)\
                    .order_by(db_models.Message.timestamp.desc())\
                    .limit(last_n)\
                    .all()
    
    history = []
    for msg in reversed(db_messages): # Reverse to get chronological order for LLM
        history.append(ChatMessageInput(role=msg.role, content=msg.content))
    return history