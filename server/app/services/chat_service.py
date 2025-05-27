from sqlalchemy.orm import Session
from ..db import models as db_models 
from ..openai_models import ChatMessageInput # For type hinting history if needed
from typing import List, Optional

def get_or_create_user(db: Session, user_identifier: str) -> db_models.User:
    user = db.query(db_models.User).filter(db_models.User.id == user_identifier).first()
    if not user:
        user = db_models.User(id=user_identifier, display_name=user_identifier)
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"INFO:     ChatService: User '{user_identifier}' created.")
    return user

def get_or_create_project(db: Session, project_name: str, description: Optional[str] = None) -> db_models.Project:
    """Gets a project by name, or creates it if it doesn't exist."""
    project = db.query(db_models.Project).filter(db_models.Project.name == project_name).first()
    if not project:
        project = db_models.Project(name=project_name, description=description)
        db.add(project)
        db.commit()
        db.refresh(project)
        print(f"INFO:     ChatService: Project '{project_name}' (ID: {project.id}) created.")
    return project

def ensure_user_linked_to_project(db: Session, user_id: str, project_id: str):
    """Ensures a user is linked to a project in the ProjectUserLink table."""
    link = db.query(db_models.ProjectUserLink).filter_by(user_id=user_id, project_id=project_id).first()
    if not link:
        link = db_models.ProjectUserLink(user_id=user_id, project_id=project_id)
        db.add(link)
        db.commit()
        print(f"INFO:     ChatService: User '{user_id}' linked to project '{project_id}'.")

def get_or_create_conversation(
    db: Session,
    conversation_id_from_owi: str, # This will be Conversation.id in our DB
    project_id: str,
    creator_user_id: str, # This is the user_identifier from OWI (e.g. {{USER_NAME}})
    title: Optional[str] = "New Chat"
) -> db_models.Conversation:
    """
    Gets a conversation by its OWI ID within a specific project, 
    or creates it if it doesn't exist.
    """
    # Ensure user and project entities exist (or are created if not)
    get_or_create_user(db, creator_user_id) 
    # Project is assumed to be created by startup logic or a project selection mechanism

    conversation = db.query(db_models.Conversation).filter(
        db_models.Conversation.id == conversation_id_from_owi,
        db_models.Conversation.project_id == project_id # Scope conversation to project
    ).first()

    if not conversation:
        conversation = db_models.Conversation(
            id=conversation_id_from_owi,
            title=title if title else "New Chat",
            project_id=project_id,
            creator_user_id=creator_user_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"INFO:     ChatService: Conversation '{conversation_id_from_owi}' created in project '{project_id}'.")
    return conversation

def add_message_to_database( # Renamed from add_message_to_db for clarity
    db: Session,
    conversation_id: str, # Our Conversation.id (which is OWI's chat_id)
    role: str, 
    content: str,
    author_user_id: Optional[str] = None, # OWI username for 'user' role; None for 'assistant'
    model_used: Optional[str] = None
) -> db_models.Message:
    """Adds a message to the specified conversation in the database."""
    if role == "user" and not author_user_id:
        print(f"WARNING:  Attempting to save user message for conversation '{conversation_id}' without author_user_id.")
        # Decide handling: raise error, use a default 'anonymous' user, or allow author_user_id to be NULL
        # For now, we allow author_user_id to be NULL in the Message model for assistant messages.
        # But for user messages, it should ideally always be present.
        # The get_or_create_user should be called before this for user messages.
        
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
    # print(f"INFO:     ChatService: Message (Role: {role}) by '{author_user_id if author_user_id else 'assistant'}' added to conv '{conversation_id}'.")
    return db_message

def get_message_history_for_llm_context(db: Session, conversation_id: str, last_n_messages: int = 10) -> List[ChatMessageInput]:
    """
    Retrieves the last N messages for a given conversation ID, formatted for LLM context.
    Returns messages in chronological order (oldest first).
    """
    # Ensure messages are ordered by timestamp correctly to get the "last N"
    db_hist_messages = db.query(db_models.Message)\
                         .filter(db_models.Message.conversation_id == conversation_id)\
                         .order_by(db_models.Message.timestamp.desc())\
                         .limit(last_n_messages)\
                         .all()
    
    # Convert to ChatMessageInput format and reverse to get chronological order for LLM
    llm_history: List[ChatMessageInput] = []
    for msg in reversed(db_hist_messages): # oldest will be first in list
        llm_history.append(ChatMessageInput(role=msg.role, content=msg.content))
    return llm_history