import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agents import RawResponsesStreamEvent
from chatkit.types import ThreadMetadata
from openai.types.responses import (
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)

from server import StarterChatServer


SEARCHING = "Webを検索しています…"
COMPOSING = "検索結果をもとに回答を作成しています…"
FAILED = "Web検索を完了できませんでした。回答を作成しています…"


def search_event(status, item_id="ws_test"):
    event_type = {
        "in_progress": ResponseWebSearchCallInProgressEvent,
        "searching": ResponseWebSearchCallSearchingEvent,
        "completed": ResponseWebSearchCallCompletedEvent,
    }[status]
    return event_type(
        type=f"response.web_search_call.{status}",
        item_id=item_id,
        output_index=0,
        sequence_number=1,
    )


def search_done(status="completed", item_id="ws_test"):
    return ResponseOutputItemDoneEvent(
        type="response.output_item.done", output_index=0, sequence_number=2,
        item={
            "type": "web_search_call", "id": item_id, "status": status,
            "action": {"type": "search", "query": "OpenAI"},
        },
    )


def message_events():
    message = {
        "type": "message", "id": "msg_test", "role": "assistant",
        "status": "in_progress", "content": [],
    }
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added", output_index=1,
            sequence_number=3, item=message,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=1, sequence_number=4,
            item={**message, "status": "completed", "content": [
                {"type": "output_text", "text": "回答です。", "annotations": []},
            ]},
        ),
    ]


class ToolProgressTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, raw_events):
        async def stream_events():
            for event in raw_events:
                yield RawResponsesStreamEvent(type="raw_response_event", data=event)

        server = StarterChatServer()
        thread = ThreadMetadata(id="thr_test", created_at=datetime.now(timezone.utc))
        result = SimpleNamespace(stream_events=stream_events)
        with patch("server.Runner.run_streamed", return_value=result):
            return [e async for e in server.respond(thread, None, {})]

    def progress(self, events):
        return [e.text for e in events if e.type == "progress_update"]

    async def test_search_status_arrives_before_answer_without_duplicates(self):
        events = await self.collect([
            search_event("in_progress"), search_event("searching"),
            search_event("searching"), search_event("completed"),
            search_done(), *message_events(),
        ])
        self.assertEqual(self.progress(events), [SEARCHING, COMPOSING])
        self.assertEqual([e.type for e in events], [
            "progress_update", "progress_update",
            "thread.item.added", "thread.item.done",
        ])
        self.assertEqual(events[-1].item.content[0].text, "回答です。")

    async def test_progress_is_delivered_while_search_is_still_running(self):
        release_search = asyncio.Event()
        progress_received = asyncio.Event()
        events = []

        async def stream_events():
            yield RawResponsesStreamEvent(
                type="raw_response_event", data=search_event("in_progress"),
            )
            await release_search.wait()
            yield RawResponsesStreamEvent(
                type="raw_response_event", data=search_event("completed"),
            )

        async def consume():
            server = StarterChatServer()
            thread = ThreadMetadata(id="thr_test", created_at=datetime.now(timezone.utc))
            async for event in server.respond(thread, None, {}):
                events.append(event)
                if event.type == "progress_update":
                    progress_received.set()

        result = SimpleNamespace(stream_events=stream_events)
        with patch("server.Runner.run_streamed", return_value=result):
            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(progress_received.wait(), timeout=2)
                self.assertEqual(self.progress(events), [SEARCHING])
                self.assertFalse(task.done())
            finally:
                release_search.set()
                await asyncio.wait_for(task, timeout=2)
        self.assertEqual(self.progress(events), [SEARCHING, COMPOSING])

    async def test_multiple_searches_wait_until_all_are_complete(self):
        events = await self.collect([
            search_event("in_progress", "ws_1"),
            search_event("in_progress", "ws_2"),
            search_event("completed", "ws_1"),
            search_event("searching", "ws_2"),
            search_event("completed", "ws_2"),
            search_event("in_progress", "ws_3"),
            search_event("completed", "ws_3"),
            *message_events(),
        ])
        self.assertEqual(self.progress(events), [
            SEARCHING, COMPOSING, SEARCHING, COMPOSING,
        ])

    async def test_unsuccessful_search_and_retry(self):
        for status in ["failed", "incomplete"]:
            with self.subTest(status=status):
                events = await self.collect([
                    search_event("in_progress", "ws_1"),
                    search_done(status, "ws_1"),
                    search_event("in_progress", "ws_2"),
                    search_event("completed", "ws_2"),
                    *message_events(),
                ])
                self.assertEqual(self.progress(events), [
                    SEARCHING, FAILED, SEARCHING, COMPOSING,
                ])

    async def test_final_tool_item_can_finish_search(self):
        events = await self.collect([
            search_event("searching"), search_done(), *message_events(),
        ])
        self.assertEqual(self.progress(events), [SEARCHING, COMPOSING])

    async def test_answer_without_search_has_no_tool_progress(self):
        events = await self.collect(message_events())
        self.assertEqual(self.progress(events), [])
        self.assertEqual([e.type for e in events], [
            "thread.item.added", "thread.item.done",
        ])


if __name__ == "__main__":
    unittest.main()
