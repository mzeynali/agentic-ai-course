"""Local SQLite CRM shaped loosely after HubSpot's contact object.

Put this behind a thin adapter if you later swap to real HubSpot - the tool
calls from the agent talk to the functions here, not to HubSpot SDKs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

DB_PATH = Path.home() / ".ops_assistant_crm.sqlite"


class Base(DeclarativeBase):
    pass


class Contact(Base):
    __tablename__ = "contacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    company: Mapped[str] = mapped_column(String(256), default="")
    title: Mapped[str] = mapped_column(String(256), default="")
    stage: Mapped[str] = mapped_column(String(64), default="lead")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    activities = relationship("Activity", back_populates="contact", cascade="all, delete-orphan")


class Activity(Base):
    __tablename__ = "activities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"))
    kind: Mapped[str] = mapped_column(String(64))
    note: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    contact = relationship("Contact", back_populates="activities")


_engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
Base.metadata.create_all(_engine)


def _to_dict(c: Contact) -> dict[str, Any]:
    return {
        "id": c.id,
        "email": c.email,
        "name": c.name,
        "company": c.company,
        "title": c.title,
        "stage": c.stage,
        "notes": c.notes,
    }


def upsert_contact(
    email: str,
    name: str,
    company: str = "",
    title: str = "",
    stage: str = "lead",
    notes: str = "",
) -> dict[str, Any]:
    with Session(_engine) as s:
        existing = s.scalar(select(Contact).where(Contact.email == email))
        if existing:
            existing.name = name or existing.name
            existing.company = company or existing.company
            existing.title = title or existing.title
            existing.stage = stage or existing.stage
            if notes:
                existing.notes = (existing.notes + "\n" + notes).strip()
            s.commit()
            return _to_dict(existing)
        c = Contact(
            email=email,
            name=name,
            company=company,
            title=title,
            stage=stage,
            notes=notes,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return _to_dict(c)


def search_contacts(query: str, limit: int = 5) -> list[dict[str, Any]]:
    q = f"%{query.lower()}%"
    with Session(_engine) as s:
        rows = (
            s.execute(
                select(Contact)
                .where(
                    (Contact.name.ilike(q))
                    | (Contact.email.ilike(q))
                    | (Contact.company.ilike(q))
                )
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [_to_dict(r) for r in rows]


def log_activity(contact_email: str, kind: str, note: str) -> dict[str, Any]:
    with Session(_engine) as s:
        contact = s.scalar(select(Contact).where(Contact.email == contact_email))
        if not contact:
            raise ValueError(f"no contact for {contact_email!r}")
        act = Activity(contact_id=contact.id, kind=kind, note=note)
        s.add(act)
        s.commit()
        s.refresh(act)
        return {"id": act.id, "contact_id": contact.id, "kind": kind, "note": note}
