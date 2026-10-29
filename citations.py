"""Render annotated URL references once, including while text is streaming."""

import re
from urllib.parse import urlsplit

from chatkit.agents import ResponseStreamConverter
from chatkit.types import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageContentPartAdded,
    AssistantMessageItem,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    URLSource,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import Field


_LINK = re.compile(
    r'(?<!!)\[([^\[\]\n]+)\]\('
    r'(?:<(https?://[^>\n]+)>|(https?://(?:[^()\s]|\([^()\s]*\))+))'
    r'(?:\s+"[^"\n]*")?\)'
)


class URLCitationAnnotation(Annotation):
    # ChatKit only keeps the insertion index; retain the original range locally
    # so we can identify citation links without deleting ordinary prose links.
    source: URLSource
    start_index: int = Field(exclude=True)
    end_index: int = Field(exclude=True)


class CitationConverter(ResponseStreamConverter):
    async def url_citation_to_annotation(
        self, citation: AnnotationURLCitation
    ) -> Annotation:
        return URLCitationAnnotation(
            source=URLSource(url=citation.url, title=citation.title),
            index=citation.end_index,
            start_index=citation.start_index,
            end_index=citation.end_index,
        )


def normalize_citations(content: AssistantMessageContent) -> AssistantMessageContent:
    """Remove only domain/URL links covered by matching citation annotations."""
    text = content.text
    ranges: set[tuple[int, int]] = set()
    for annotation in content.annotations:
        if not isinstance(annotation, URLCitationAnnotation):
            continue
        if not 0 <= annotation.start_index < annotation.end_index <= len(text):
            continue
        source = annotation.source
        if not isinstance(source, URLSource):
            continue
        try:
            hostname = urlsplit(source.url).hostname
        except ValueError:
            continue
        if not hostname:
            continue
        labels = {hostname, hostname.removeprefix("www."), source.url.rstrip("/")}
        for match in _LINK.finditer(text, annotation.start_index, annotation.end_index):
            label, angle_url, url = match.groups()
            if (angle_url or url) == source.url and label.strip().rstrip("/") in labels:
                ranges.add(match.span())

    # A parenthesized group of citation links should not leave empty brackets or
    # separators behind. Preserve all punctuation outside the citation group.
    groups: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if groups and re.fullmatch(r"[\s,;]*", text[groups[-1][1]:start]):
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    removals = []
    for start, end in groups:
        left, right = start, end
        while left > 0 and text[left - 1] in " \t":
            left -= 1
        while right < len(text) and text[right] in " \t":
            right += 1
        if left > 0 and text[left - 1] == "(":
            if right < len(text) and text[right] == ")":
                start, end = left - 1, right + 1
            elif right == len(text):
                # The closing parenthesis can arrive in the next text delta.
                start, end = left - 1, right
        removals.append((start, end))

    def remap(index: int | None) -> int | None:
        if index is None:
            return None
        return index - sum(max(0, min(index, end) - start) for start, end in removals)

    for start, end in reversed(removals):
        text = text[:start] + text[end:]
    return AssistantMessageContent(
        text=text,
        annotations=[
            Annotation(source=a.source, index=remap(a.index))
            for a in content.annotations
        ],
    )


class CitationRenderer:
    """Keep original content for offsets, and send normalized copies to ChatKit."""

    def __init__(self) -> None:
        self.parts: dict[tuple[str, int], AssistantMessageContent] = {}

    def process(self, event: ThreadStreamEvent) -> ThreadStreamEvent:
        if event.type in {"thread.item.added", "thread.item.done"}:
            if not isinstance(event.item, AssistantMessageItem):
                return event
            for index, content in enumerate(event.item.content):
                key = (event.item.id, index)
                if event.type == "thread.item.done":
                    self.parts.pop(key, None)
                else:
                    self.parts[key] = content.model_copy(deep=True)
            return event.model_copy(update={"item": event.item.model_copy(update={
                "content": [normalize_citations(c) for c in event.item.content],
            })})

        if event.type != "thread.item.updated":
            return event
        update = event.update
        if not update.type.startswith("assistant_message.content_part."):
            return event
        key = (event.item_id, update.content_index)
        if update.type == "assistant_message.content_part.added":
            self.parts[key] = update.content.model_copy(deep=True)
            return event.model_copy(update={"update": update.model_copy(update={
                "content": normalize_citations(update.content),
            })})

        content = self.parts.get(key)
        if content is None:
            return event
        if update.type == "assistant_message.content_part.done":
            # ChatKit 1.6.5 emits text.done with an empty annotations list.
            # Keep the citations already received until output_item.done arrives.
            content.text = update.content.text
            if update.content.annotations:
                content.annotations = list(update.content.annotations)
            return event.model_copy(update={"update": update.model_copy(update={
                "content": normalize_citations(content),
            })})

        previous = normalize_citations(content)
        if update.type == "assistant_message.content_part.text_delta":
            content.text += update.delta
            rendered = normalize_citations(content)
            if rendered.text.startswith(previous.text) and rendered.annotations == previous.annotations:
                return event.model_copy(update={"update": update.model_copy(update={
                    "delta": rendered.text[len(previous.text):],
                })})
        elif update.type == "assistant_message.content_part.annotation_added":
            content.annotations.insert(update.annotation_index, update.annotation)
            rendered = normalize_citations(content)
        else:
            return event

        # Replacing the content part also removes a link already streamed before
        # its annotation arrived. The "added" event keeps the part streaming.
        return ThreadItemUpdatedEvent(
            item_id=event.item_id,
            update=AssistantMessageContentPartAdded(
                content_index=update.content_index,
                content=rendered,
            ),
        )
