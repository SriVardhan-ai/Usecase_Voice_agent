import asyncio

from dotenv import load_dotenv

from livekit import agents
from langchain_core.messages import ToolMessage
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, inference, room_io
from livekit.plugins import langchain as lk_langchain

from config import settings
from langchain_agent import lauki_agent, qdrant, embedder

_ = load_dotenv()


def _is_tool_token(item: object) -> bool:
    """True for the (ToolMessage, metadata) pairs astream emits from the tools node."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class LaukiAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions="You are the evernorth health care and services customer support voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(lauki_agent)),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    def _warm_up() -> None:
        embedder.encode("warm up")
        qdrant.get_collection(settings.collection_name)

    # load the embedding model + open Qdrant in the background so the first question is fast
    asyncio.create_task(asyncio.to_thread(_warm_up))

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(
            model="inworld/inworld-tts-2",
            voice="Ashley",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.5,
                "max_delay": 6.0,
            },
        ),
    )

    await session.start(
        agent=LaukiAgent(),
        room=ctx.room,
        # show text as soon as the LLM produces it instead of pacing it to the audio.
        # Text typed in the website chat box (topic "lk.chat") is on by default.
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(sync_transcription=False),
        ),
    )

    await session.generate_reply(user_input="Welcome to evernorth health care and services.I am your customer assistant and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(server)
