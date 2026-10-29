import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agents import RawResponsesStreamEvent
from chatkit.types import Annotation, AssistantMessageContent, FileSource, ThreadMetadata
from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from citations import CitationConverter, normalize_citations
from server import CitationCompatibleStream, StarterChatServer


class CitationTests(unittest.IsolatedAsyncioTestCase):
    async def citation(self, text, link, url, title="Source"):
        from openai.types.responses.response_output_text import AnnotationURLCitation

        start = text.index(link)
        return await CitationConverter().url_citation_to_annotation(
            AnnotationURLCitation(
                type="url_citation", url=url, title=title,
                start_index=start, end_index=start + len(link),
            )
        )

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

    async def test_citation_group_is_removed_and_all_offsets_are_preserved(self):
        apple = "[apple.com](https://www.apple.com/news/)"
        other = "[example.com](https://example.com/release_(2026))"
        text = f"発売されました。 ({apple}, {other})\n続き。"
        content = AssistantMessageContent(text=text, annotations=[
            await self.citation(text, apple, "https://www.apple.com/news/"),
            await self.citation(text, other, "https://example.com/release_(2026)"),
            Annotation(source=FileSource(title="Document", filename="doc.pdf"), index=len(text)),
        ])
        rendered = normalize_citations(content)
        self.assertEqual(rendered.text, "発売されました。 \n続き。")
        self.assertEqual([a.index for a in rendered.annotations], [9, 9, len(rendered.text)])
        self.assertEqual(content.text, text)
        self.assertNotIn("start_index", rendered.model_dump_json())

    async def test_descriptive_and_unannotated_links_are_kept(self):
        url = "https://www.apple.com/news/"
        descriptive = f"[Appleの発表]({url})"
        unannotated = f"[apple.com]({url})"
        text = f"詳しくは{descriptive}を参照。\n{unannotated}"
        content = AssistantMessageContent(text=text, annotations=[
            await self.citation(text, descriptive, url),
        ])
        self.assertEqual(normalize_citations(content).text, text)

    async def test_non_link_citation_and_mismatched_url_are_kept(self):
        text = "Apple released a product. [apple.com](https://www.apple.com/news/)"
        content = AssistantMessageContent(text=text, annotations=[
            await self.citation(text, "Apple released a product.", "https://www.apple.com/news/"),
            await self.citation(text, text, "https://www.apple.com/other/"),
        ])
        self.assertEqual(normalize_citations(content).text, text)

    async def test_real_sdk_stream_never_duplicates_a_received_citation(self):
        url = "https://www.apple.com/news/"
        link = f"[apple.com]({url})"
        first = f"Released. ({link}"
        final = first + ")\nMore text."
        citation = {
            "type": "url_citation", "url": url, "title": "Apple",
            "start_index": final.index(link), "end_index": final.index(link) + len(link),
        }
        common = dict(item_id="msg_test", output_index=0, content_index=0)
        raw_events = [
            ResponseContentPartAddedEvent(
                type="response.content_part.added", sequence_number=1, **common,
                part={"type": "output_text", "text": "", "annotations": []},
            ),
            ResponseTextDeltaEvent(type="response.output_text.delta", sequence_number=2, delta=first, logprobs=[], **common),
            ResponseOutputTextAnnotationAddedEvent(
                type="response.output_text.annotation.added", sequence_number=3,
                annotation_index=0, annotation=citation, **common,
            ),
            ResponseTextDeltaEvent(type="response.output_text.delta", sequence_number=4, delta=")\nMore text.", logprobs=[], **common),
            ResponseTextDoneEvent(type="response.output_text.done", sequence_number=5, text=final, logprobs=[], **common),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done", sequence_number=6, output_index=0,
                item={
                    "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": final, "annotations": [citation]}],
                },
            ),
        ]

        async def stream_events():
            for raw in raw_events:
                yield RawResponsesStreamEvent(type="raw_response_event", data=raw)

        server = StarterChatServer()
        thread = ThreadMetadata(id="thr_test", created_at=datetime.now(timezone.utc))
        with patch("server.Runner.run_streamed", return_value=SimpleNamespace(stream_events=stream_events)):
            events = [e async for e in server.respond(thread, None, {})]

        # Reconstruct the actual client/server content state after each event.
        content = None
        for event in events:
            if event.type == "thread.item.done":
                content = event.item.content[0]
            elif hasattr(event.update, "content"):
                content = event.update.content.model_copy(deep=True)
            else:
                content.text += event.update.delta
            if content.annotations:
                self.assertNotIn(link, content.text)
                self.assertEqual(content.annotations[0].source.url, url)
                self.assertEqual(content.annotations[0].index, len("Released. "))

        self.assertEqual(content.text, "Released. \nMore text.")
        self.assertEqual(len(content.annotations), 1)
        self.assertEqual(events[-2].update.content.annotations, content.annotations)


if __name__ == "__main__":
    unittest.main()
