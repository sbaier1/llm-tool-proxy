from flask import Flask, request, jsonify, Response
import requests
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from typing import Optional, List, Mapping, Any, Generator, Union
import logging
import time
import json
import os

# tools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ShellTool

# The actual LLM we will call to.
UPSTREAM_LLM = os.getenv("UPSTREAM_LLM")
if UPSTREAM_LLM is None:
    raise KeyError("You must configure the environment variable UPSTREAM_LLM. Must be a HTTP(s) URL.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Define a simple calculator tool
def calculator_tool(text):
    try:
        return str(eval(text))
    except Exception as e:
        return f"Error in calculation: {e}"


python_repl = PythonREPL()


# Define the validation function
def validate_shell_input(input_data: str):
    # List of forbidden keywords/commands
    forbidden_keywords = [" delete ", "rm ", "mv "]

    # Check if the input contains any forbidden keywords
    for keyword in forbidden_keywords:
        if keyword in input_data:
            raise ValueError(f"Disallowed command detected: '{keyword.strip()}'")


# Subclass the ShellTool to add validation
class CustomShellTool(ShellTool):
    def run(self,
            tool_input: Union[str, dict[str, Any]],
            verbose: Optional[bool] = None,
            start_color: Optional[str] = 'green',
            color: Optional[str] = 'green',
            callbacks: Optional[Any] = None,
            *,
            tags: Optional[list[str]] = None,
            metadata: Optional[dict[str, Any]] = None,
            run_name: Optional[str] = None,
            run_id: Optional[Any] = None,
            config: Optional[Any] = None,
            tool_call_id: Optional[str] = None,
            **kwargs: Any) -> Any:

        # Extract the actual shell command input (assuming it's a string)
        if isinstance(tool_input, str):
            validate_shell_input(tool_input)
        elif isinstance(tool_input, dict) and "input" in tool_input:
            validate_shell_input(tool_input["input"])
        else:
            raise ValueError("Unsupported input format for ShellTool.")

        # Proceed with the original execution of ShellTool
        return super().run(
            tool_input,
            verbose=verbose,
            start_color=start_color,
            color=color,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_id=run_id,
            config=config,
            tool_call_id=tool_call_id,
            **kwargs
        )


tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for when you need to perform math calculations. Write expressions as Python code."
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    ),
    CustomShellTool(),
    DuckDuckGoSearchResults(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
]


# Custom LLM that sends requests to the downstream API
class CustomLLM(LLM):
    endpoint_url: str = UPSTREAM_LLM

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": "You are an assistant that uses tools to answer questions."},
                {"role": "user", "content": prompt}
            ],
            # TODO just pass in params from initial query
            "max_tokens": 4000,
            "temperature": 0.15,
        }
        if stop is not None:
            payload['stop'] = stop
        response = requests.post(self.endpoint_url, json=payload)
        response_json = response.json()
        # Extract the assistant's reply
        return response_json['choices'][0]['message']['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"endpoint_url": self.endpoint_url}

    @property
    def _llm_type(self) -> str:
        return "custom"


# Function to simulate streaming by yielding chunks
def generate_streamed_response(content: str) -> Generator[str, None, None]:
    buffer = ""
    for char in content:
        buffer += char
        # Define a chunk size or send per character
        if len(buffer) >= 5:
            chunk = {
                "choices": [
                    {
                        "delta": {"content": buffer},
                        "index": 0,
                        "finish_reason": None
                    }
                ],
                "id": "chatcmpl-KrQgb670aQeEUSzf0TKG3GQA2HuWwHBg",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "custom",
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            buffer = ""
            time.sleep(0.05)  # Simulate delay between chunks
    if buffer:
        # Send any remaining characters
        chunk = {
            "choices": [
                {
                    "delta": {"content": buffer},
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "id": "chatcmpl-KrQgb670aQeEUSzf0TKG3GQA2HuWwHBg",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "custom",
        }
        yield f"data: {json.dumps(chunk)}\n\n"


# Route for OpenAI completion-style API
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        messages = data.get('messages', [])
        stream = data.get('stream', False)

        # Extract last user message and build memory
        last_user_message = ""
        conversation_history = []

        # Assuming messages are in order
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            if role == 'user':
                conversation_history.append({'role': 'user', 'content': content})
                last_user_message = content  # Update last_user_message to the latest
            elif role == 'assistant':
                conversation_history.append({'role': 'assistant', 'content': content})

        # Initialize memory and populate it with previous conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory.chat_memory.messages = []

        # Exclude the last user message from the memory (since it will be 'input')
        if conversation_history and conversation_history[-1]['role'] == 'user':
            last_user_message = conversation_history.pop()['content']

        # Build the messages for the memory
        for msg in conversation_history:
            if msg['role'] == 'user':
                memory.chat_memory.add_user_message(msg['content'])
            elif msg['role'] == 'assistant':
                memory.chat_memory.add_ai_message(msg['content'])

        # Initialize the custom LLM and agent with memory
        llm = CustomLLM()
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory
        )

        # Process the prompt through the agent
        try:
            response_content = agent.run(input=last_user_message)
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing the completion request: {str(e)}")
            return jsonify({'error': str(e)}), 500

        if stream:
            def streamed():
                for chunk in generate_streamed_response(response_content):
                    yield chunk

            return Response(streamed(), mimetype='text/event-stream')
        else:
            # Construct the response in OpenAI API format
            response_payload = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "custom",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content,
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            return jsonify(response_payload)

    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing the completion request: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5002)
