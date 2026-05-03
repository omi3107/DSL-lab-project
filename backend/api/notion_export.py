"""
notion_export.py — Export meeting insights as formatted Notion pages.

Converts structured meeting data into Notion blocks (headings, bullets,
tables, dividers, callouts) and creates a readable page via the Notion API.
"""

import os
import httpx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

NOTION_API_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _get_headers():
    token = os.getenv("NOTION_API_KEY", "")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def _get_database_id():
    return os.getenv("NOTION_DATABASE_ID", "")


# ── Block Builders ───────────────────────────────────────────────

def _heading2(text: str, emoji: str = "") -> dict:
    prefix = f"{emoji} " if emoji else ""
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": f"{prefix}{text}"}}]
        },
    }


def _heading3(text: str, emoji: str = "") -> dict:
    prefix = f"{emoji} " if emoji else ""
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {
            "rich_text": [{"type": "text", "text": {"content": f"{prefix}{text}"}}]
        },
    }


def _bullet(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def _paragraph(text: str, bold: bool = False) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": text},
                    "annotations": {"bold": bold},
                }
            ]
        },
    }


def _rich_paragraph(segments: list) -> dict:
    """Create a paragraph with mixed formatting. Each segment: (text, bold)."""
    rich_text = []
    for text, bold in segments:
        rich_text.append({
            "type": "text",
            "text": {"content": text},
            "annotations": {"bold": bold},
        })
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich_text},
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _callout(text: str, emoji: str = "💡") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": emoji},
            "rich_text": [{"type": "text", "text": {"content": text}}],
        },
    }


def _table_row(cells: list) -> dict:
    """Build a table row. Each cell is a string."""
    return {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [[{"type": "text", "text": {"content": c}}] for c in cells]
        },
    }


def _table(rows: list, col_count: int) -> dict:
    """Build a table block with header. rows = list of cell-lists."""
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": True,
            "has_row_header": False,
            "children": [_table_row(r) for r in rows],
        },
    }


# ── Page Builder ─────────────────────────────────────────────────

def _build_page_properties(data: dict) -> dict:
    """Build Notion page properties for the database."""
    title = data.get("title", "Meeting Analysis")
    return {
        "Name": {
            "title": [{"text": {"content": title}}]
        },
    }


def _build_page_blocks(data: dict) -> list:
    """Convert meeting insights into Notion content blocks."""
    blocks = []
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    score = data.get("score", 0)
    tags = data.get("tags", [])
    participants = data.get("participants", [])
    decisions = data.get("decisions", [])
    tasks = data.get("tasks", [])
    deadlines = data.get("deadlines", [])
    issues = data.get("issues", [])
    general = data.get("general", [])
    title = data.get("title", "Meeting Analysis")

    # ── Header Info ──
    score_label = (
        "Highly Productive" if score >= 80
        else "Productive" if score >= 70
        else "Moderate" if score >= 50
        else "Average" if score >= 40
        else "Needs Improvement"
    )
    tags_str = "  •  ".join(tags) if tags else "—"

    blocks.append(_heading2(title, "📌"))
    blocks.append(_rich_paragraph([
        ("📅 ", False), (f"{date_str}  {time_str}", False),
        ("   |   🧠 Score: ", False), (f"{score}/100 ({score_label})", True),
        ("   |   🏷️ ", False), (tags_str, False),
    ]))
    blocks.append(_divider())

    # ── Participants ──
    blocks.append(_heading3("Participants", "👥"))
    if participants:
        for p in participants:
            task_count = sum(1 for t in tasks if isinstance(t, dict) and t.get("who") == p)
            blocks.append(_bullet(f"{p}  ({task_count} task{'s' if task_count != 1 else ''})"))
    else:
        blocks.append(_paragraph("No participants detected."))
    blocks.append(_divider())

    # ── 1-line Summary ──
    blocks.append(_heading3("Summary", "💬"))
    summary_parts = []
    if decisions:
        summary_parts.append(decisions[0])
    if tasks and isinstance(tasks[0], dict):
        summary_parts.append(f"{tasks[0].get('who', '?')} → {tasks[0].get('task', '?')}")
    if not summary_parts and general:
        summary_parts.append(general[0])
    blocks.append(_callout(", ".join(summary_parts) if summary_parts else "No summary available.", "💡"))
    blocks.append(_divider())

    # ── Decisions ──
    blocks.append(_heading2("Decisions", "✅"))
    if decisions:
        for d in decisions:
            blocks.append(_bullet(str(d)))
    else:
        blocks.append(_paragraph("No decisions recorded."))

    # ── Tasks (as table) ──
    blocks.append(_heading2("Tasks", "📌"))
    if tasks:
        table_rows = [["Assignee", "Task", "Due Date"]]
        for t in tasks:
            if isinstance(t, dict):
                table_rows.append([
                    t.get("who", "Unassigned"),
                    t.get("task", "—"),
                    t.get("by", "—") or "—",
                ])
            else:
                table_rows.append(["Unassigned", str(t), "—"])
        blocks.append(_table(table_rows, 3))
    else:
        blocks.append(_paragraph("No tasks assigned."))

    # ── Deadlines ──
    blocks.append(_heading2("Deadlines", "📅"))
    if deadlines:
        for d in deadlines:
            text = d.get("description", str(d)) if isinstance(d, dict) else str(d)
            blocks.append(_bullet(text))
    else:
        blocks.append(_paragraph("No deadlines set."))

    # ── Issues ──
    blocks.append(_heading2("Issues", "⚠️"))
    if issues:
        for iss in issues:
            blocks.append(_callout(str(iss), "🔴"))
    else:
        blocks.append(_paragraph("No issues detected."))

    # ── General Discussion ──
    blocks.append(_heading2("General Discussion", "🗣️"))
    if general:
        for g in general:
            blocks.append(_bullet(str(g)))
    else:
        blocks.append(_paragraph("No general discussion points."))

    # ── Footer ──
    blocks.append(_divider())
    blocks.append(_rich_paragraph([
        ("📊 Intelligence Score: ", True),
        (f"{score}/100", False),
        (f"  —  {score_label}", False),
    ]))
    blocks.append(_rich_paragraph([
        ("📁 Decisions: ", True), (f"{len(decisions)}", False),
        ("  |  Tasks: ", True), (f"{len(tasks)}", False),
        ("  |  Deadlines: ", True), (f"{len(deadlines)}", False),
        ("  |  Issues: ", True), (f"{len(issues)}", False),
        ("  |  General: ", True), (f"{len(general)}", False),
    ]))

    return blocks


# ── API Calls ────────────────────────────────────────────────────

async def check_notion_connection() -> dict:
    """Check if Notion credentials are configured and valid."""
    token = os.getenv("NOTION_API_KEY", "")
    db_id = _get_database_id()
    if not token or not db_id:
        return {"connected": False, "reason": "Missing NOTION_API_KEY or NOTION_DATABASE_ID"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{NOTION_API_URL}/databases/{db_id}",
                headers=_get_headers(),
            )
            if resp.status_code == 200:
                return {"connected": True, "database_title": resp.json().get("title", [{}])[0].get("plain_text", "Unknown")}
            else:
                return {"connected": False, "reason": f"Notion API error: {resp.status_code}"}
    except Exception as e:
        return {"connected": False, "reason": str(e)}


async def export_to_notion(meeting_data: dict) -> dict:
    """Create a Notion page with formatted meeting insights."""
    db_id = _get_database_id()
    if not db_id:
        raise ValueError("NOTION_DATABASE_ID not configured")

    properties = _build_page_properties(meeting_data)
    children = _build_page_blocks(meeting_data)

    # Notion API limits children to 100 blocks per request
    first_batch = children[:100]
    remaining = children[100:]

    payload = {
        "parent": {"database_id": db_id},
        "properties": properties,
        "children": first_batch,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{NOTION_API_URL}/pages",
            headers=_get_headers(),
            json=payload,
        )

        if resp.status_code not in (200, 201):
            error_body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            raise RuntimeError(
                f"Notion API error {resp.status_code}: {error_body.get('message', resp.text[:200])}"
            )

        page = resp.json()
        page_id = page["id"]
        page_url = page.get("url", "")

        # Append remaining blocks if any
        if remaining:
            for i in range(0, len(remaining), 100):
                batch = remaining[i : i + 100]
                await client.patch(
                    f"{NOTION_API_URL}/blocks/{page_id}/children",
                    headers=_get_headers(),
                    json={"children": batch},
                )

    logger.info("Notion page created: %s", page_url)
    return {"url": page_url, "page_id": page_id}
