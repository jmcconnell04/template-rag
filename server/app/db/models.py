from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func # for server_default with timezone
from .database import Base
import uuid # For generating default IDs
import datetime # For timezone aware datetime

def generate_uuid_str():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    # Using the identifier from OWI (e.g., {{USER_NAME}}) as the primary key.
    id = Column(String, primary_key=True, index=True) 
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    projects = relationship("ProjectUserLink", back_populates="user")
    created_conversations = relationship("Conversation", back_populates="creator_user", foreign_keys="[Conversation.creator_user_id]")
    messages_authored = relationship("Message", back_populates="author_user", foreign_keys="[Message.author_user_id]")

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=generate_uuid_str)
    name = Column(String, unique=True, index=True, nullable=False) # e.g., "WORKSPACE_NAME Project"
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    users = relationship("ProjectUserLink", back_populates="project")
    conversations = relationship("Conversation", back_populates="project")

class ProjectUserLink(Base):
    __tablename__ = "project_user_links"
    project_id = Column(String, ForeignKey("projects.id"), primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    linked_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="projects")
    project = relationship("Project", back_populates="users")

class Conversation(Base):
    __tablename__ = "conversations"
    # Using OpenWebUI's chat_id as the primary key
    id = Column(String, primary_key=True, index=True) 
    title = Column(String, default="New Chat")
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    creator_user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    project = relationship("Project", back_populates="conversations")
    creator_user = relationship("User", back_populates="created_conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.timestamp.asc()")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=generate_uuid_str) # Own ID for messages
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    # author_user_id for 'user' role messages will be the OWI username.
    # For 'assistant' role, it can be NULL.
    author_user_id = Column(String, ForeignKey("users.id"), nullable=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_used = Column(String, nullable=True) # For assistant messages

    conversation = relationship("Conversation", back_populates="messages")
    author_user = relationship("User", back_populates="messages_authored")