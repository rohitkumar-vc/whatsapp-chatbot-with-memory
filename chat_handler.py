import asyncio
from datetime import datetime
from fastapi import BackgroundTasks
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

# Assuming your setup and imports
from config import app_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from memory import Memory, config

# Instantiate Mem0 globally (or pass it in)
mem0_client = Memory.from_config(config)

class ChatHandler:
    def __init__(self, background_tasks: BackgroundTasks, user_id: str):
        """
        Initializes the chat handler for a specific WhatsApp user.
        """
        self.background_tasks = background_tasks
        self.user_id = user_id
        self.messages = [] 
        self.last_active = datetime.now()
        
        # Async lock to prevent race conditions during list mutation
        self.compaction_lock = asyncio.Lock()
        self.is_compacting = False

        # Initialize the LLM using your provided configuration
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            api_key=app_settings.GOOGLE_API_KEY,
            vertexai=app_settings.VERTEX_AI,
        )

        # 1. Define the Memory Retrieval Tool for the ReAct Agent
        @tool
        def retrieve_memory(query: str) -> str:
            """Search the user's long-term memory for past facts, preferences, and context."""
            # Mem0 search using the specific user_id
            results = mem0_client.search(query, user_id=self.user_id)
            if not results.get("results"):
                return "No relevant memories found."
            return "\n".join([r["memory"] for r in results["results"]])

        # 2. Create the ReAct Agent using LangChain's create_agent
        self.agent = create_agent(
            model=self.llm,
            tools=[retrieve_memory],
            system_prompt=(
                "You are a helpful and personalized WhatsApp companion. "
                "Use the retrieve_memory tool if you need context about the user's past."
            )
        )

    async def handle_message(self, user_input: str) -> str:
        """
        Main entry point for incoming WhatsApp messages.
        """
        # Record activity time to manage the session pause (idle trigger)
        self.last_active = datetime.now()
        
        # Append incoming message
        self.messages.append(HumanMessage(content=user_input))

        # Invoke the ReAct Agent. 
        response = await self.agent.ainvoke({"messages": self.messages})
        ai_reply = response["messages"][-1].content
        
        self.messages.append(AIMessage(content=ai_reply))

        # Check Cap Limit (50) - Trigger background compaction immediately if needed
        if len(self.messages) >= 50:
            self.background_tasks.add_task(self._process_memory_update, is_idle=False)
        else:
            # Schedule the Idle Check (5 minutes)
            self.background_tasks.add_task(self._check_idle_and_update, self.last_active)

        return ai_reply

    async def _check_idle_and_update(self, trigger_time: datetime):
        """
        Waits 5 minutes. If the user hasn't sent a new message, triggers memory update.
        """
        await asyncio.sleep(300) # 300 seconds = 5 mins
        
        # If last_active hasn't changed, the session has officially paused
        if self.last_active == trigger_time:
            await self._process_memory_update(is_idle=True)

    async def _process_memory_update(self, is_idle: bool):
        """
        The core asynchronous compaction and memory extraction engine.
        """
        async with self.compaction_lock:
            # Prevent multiple compaction tasks from running simultaneously
            if self.is_compacting:
                return
            
            msg_count = len(self.messages)
            
            # Rule 1: Idle update only if we have > 20 messages
            if is_idle and msg_count <= 20:
                return
            # Rule 2: Cap update hits at 50
            if not is_idle and msg_count < 50:
                return

            self.is_compacting = True
            
            # Calculate how many messages to slice out.
            # If hit cap (50), summarize first 40. 
            # If idle and > 20, summarize all but the last 10 to keep immediate context.
            slice_index = 40 if not is_idle else (msg_count - 10)
            
            # Take a snapshot of the messages to process
            messages_to_process = self.messages[:slice_index]

        try:
            # Format messages for the summarization and extraction tasks
            chat_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages_to_process])

            # 1. Generate Context Summary for the short-term list (Fast LLM Call)
            summary_prompt = f"Summarize this chat history concisely to retain conversation context:\n{chat_text}"
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary_text = summary_response.content

            # 2. Extract Facts for Long-Term Memory (Mem0)
            # You can pass the raw chat directly to Mem0 for processing
            mem0_formatted = [{"role": msg.type, "content": msg.content} for msg in messages_to_process]
            mem0_client.add(mem0_formatted, user_id=self.user_id)

            # 3. Safely splice the list using the lock
            async with self.compaction_lock:
                # Replace the exact slice we took with 1 single summary AIMessage.
                # Any new messages that arrived during processing remain untouched at the end of the list.
                summary_msg = AIMessage(content=f"[Previous Context Summary]: {summary_text}")
                self.messages = [summary_msg] + self.messages[slice_index:]

        finally:
            # Always release the lock flag
            async with self.compaction_lock:
                self.is_compacting = False