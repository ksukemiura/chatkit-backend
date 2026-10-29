import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agents import RawResponsesStreamEvent
from chatkit.types import ThreadMetadata
from openai.types.responses import ResponseOutputTextAnnotationAddedEvent

from server import CitationCompatibleStream, StarterChatServer


class CitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_sdk_citation_reaches_chatkit(self):
        # Use the actual streaming SDK type, not the similarly named final
        # output type: the two classes caused a production validation error.
        raw = RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputTextAnnotationAddedEvent(
                type="response.output_text.annotation.added",
                annotation={
                    "type": "url_citation",
                    "url": "https://openai.com/",
                    "title": "OpenAI",
                    "start_index": 0,
                    "end_index": 6,
                },
                content_index=0,
                item_id="msg_test",
                annotation_index=0,
                output_index=0,
                sequence_number=1,
            ),
        )

        async def stream_events():
            yield raw

        result = SimpleNamespace(stream_events=stream_events)
        server = StarterChatServer()
        thread = ThreadMetadata(
            id="thr_test", created_at=datetime.now(timezone.utc)
        )
        with patch("server.Runner.run_streamed", return_value=result):
            events = [e async for e in server.respond(thread, None, {})]

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.type, "thread.item.updated")
        self.assertEqual(event.update.type, "assistant_message.content_part.annotation_added")
        self.assertEqual(event.update.annotation.source.url, "https://openai.com/")
        self.assertEqual(event.update.annotation.source.title, "OpenAI")
        self.assertEqual(event.update.annotation.index, 6)
        self.assertFalse(isinstance(raw.data.annotation, dict))

    async def test_non_annotation_events_are_forwarded_unchanged(self):
        original = SimpleNamespace(type="agent_updated_stream_event")

        async def stream_events():
            yield original

        result = SimpleNamespace(stream_events=stream_events)
        events = [e async for e in CitationCompatibleStream(result).stream_events()]
        self.assertEqual(events, [original])


if __name__ == "__main__":
    unittest.main()
