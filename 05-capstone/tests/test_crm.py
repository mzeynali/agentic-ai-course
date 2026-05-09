"""Round-trip test for the local CRM store.

Uses an in-memory SQLite by monkeypatching the engine so the test never
touches your real CRM file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sqlalchemy import create_engine

from integrations import crm_store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch):
    # Swap the module-global engine to an in-memory DB just for this test.
    engine = create_engine("sqlite:///:memory:", future=True)
    crm_store.Base.metadata.create_all(engine)
    monkeypatch.setattr(crm_store, "_engine", engine)


def test_upsert_creates_then_updates():
    c = crm_store.upsert_contact(
        email="sara@acme.com", name="Sara Chen", company="Acme", notes="Met at conf"
    )
    assert c["email"] == "sara@acme.com"

    c2 = crm_store.upsert_contact(
        email="sara@acme.com", name="Sara Chen", notes="Followed up on Apr 23"
    )
    assert "Followed up" in c2["notes"]
    assert "Met at conf" in c2["notes"]  # preserved from before


def test_search_contacts_by_name_email_company():
    crm_store.upsert_contact(email="a@x.com", name="Alice", company="Globex")
    crm_store.upsert_contact(email="b@y.com", name="Bob", company="Initech")

    assert len(crm_store.search_contacts("alice")) == 1
    assert len(crm_store.search_contacts("y.com")) == 1
    assert len(crm_store.search_contacts("Initech")) == 1
    assert crm_store.search_contacts("nobody") == []


def test_log_activity_requires_existing_contact():
    crm_store.upsert_contact(email="alice@x.com", name="Alice")
    act = crm_store.log_activity("alice@x.com", "email", "sent intro")
    assert act["kind"] == "email"

    with pytest.raises(ValueError):
        crm_store.log_activity("ghost@x.com", "email", "???")
