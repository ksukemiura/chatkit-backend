"""Report hosted tool activity through ChatKit's transient progress UI."""

from collections.abc import AsyncIterator
from typing import Any

from agents import RunResultStreaming
from agents.stream_events import StreamEvent
from chatkit.agents import AgentContext
from chatkit.types import ProgressUpdateEvent


class ToolProgressStream:
    def __init__(self, result: RunResultStreaming, context: AgentContext) -> None:
        self.result = result
        self.context = context
        self.active_searches: set[str] = set()
        self.search_failed = False
        self.last_text: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        async for event in self.result.stream_events():
            if event.type == "raw_response_event":
                text = self._progress_text(event.data)
                if text is not None and text != self.last_text:
                    await self.context.stream(ProgressUpdateEvent(text=text))
                    self.last_text = text
            yield event

    def _progress_text(self, event: Any) -> str | None:
        if event.type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
        }:
            if not self.active_searches:
                self.search_failed = False
            self.active_searches.add(event.item_id)
            return "Webを検索しています…"

        if event.type == "response.web_search_call.completed":
            item_id = event.item_id
        elif (
            event.type == "response.output_item.done"
            and event.item.type == "web_search_call"
        ):
            item_id = event.item.id
            if item_id not in self.active_searches:
                return None
            self.search_failed |= event.item.status in {"failed", "incomplete"}
        else:
            return None

        self.active_searches.discard(item_id)
        if self.active_searches:
            return None
        if self.search_failed:
            return "Web検索を完了できませんでした。回答を作成しています…"
        return "検索結果をもとに回答を作成しています…"
