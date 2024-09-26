import json
import logging
import os
import subprocess
import time
from typing import Optional, List, Mapping, Any, Generator

import requests
from flask import Flask, request, jsonify, Response
from langchain.agents import Tool, AgentExecutor, ConversationalAgent
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
# tools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.utilities import PythonREPL

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


# Bash Command Execution Tool
def bash_command_tool(text):
    forbidden_commands = ['rm ', 'ln ', 'mv ']

    # Check for forbidden commands
    for command in forbidden_commands:
        if command in text:
            return f"Error: Command '{command}' is not allowed."

    try:
        # Run the command and capture the output
        result = subprocess.run(text, shell=True, capture_output=True, text=True)
        # Truncate output if it exceeds 5000 characters
        max_output_length = 5000
        if isinstance(result, str) and len(result) > max_output_length:
            result = result[:max_output_length] + "\n[Output truncated: exceeded 5000 characters]"
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error executing bash command: {e}"


python_repl = PythonREPL()


# Define the validation function
def validate_shell_input(input_data: str):
    # List of forbidden keywords/commands
    forbidden_keywords = [" delete ", "rm ", "mv "]

    # Check if the input contains any forbidden keywords
    for keyword in forbidden_keywords:
        if keyword in input_data:
            raise ValueError(f"Disallowed command detected: '{keyword.strip()}'")



# Custom LLM that sends requests to the downstream API
class CustomLLM(LLM):
    endpoint_url: str = UPSTREAM_LLM

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": "You are an assistant that can use tools to answer questions. "
                                              "Think whether you need to use a tool before you execute a tool."},
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


# Nested agent tool function
def nested_agent_tool(input_text: str) -> str:
    print(f"Forking an agent for prompt {input_text}")
    # Create a new memory for the nested agent
    nested_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Initialize the nested agent
    nested_llm = CustomLLM()
    nested_agent_chain = ConversationalAgent.from_llm_and_tools(
        llm=nested_llm,
        tools=tools,  # Use the same tools
        verbose=True,
    )
    nested_agent = AgentExecutor.from_agent_and_tools(
        agent=nested_agent_chain,
        tools=tools,
        memory=nested_memory,
        verbose=True,
    )
    # Run the nested agent with the input text
    try:
        return nested_agent.run(input=input_text)
    except Exception as e:
        logger.exception(f"Error in nested agent: {str(e)}")
        return f"Error in nested agent: {str(e)}"


# Initialize tools
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
    DuckDuckGoSearchResults(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    Tool(
        name="Nested Agent",
        func=nested_agent_tool,
        description="Use this tool to delegate a sub-task to another agent. "
                    "Input should be the task you want the nested agent to perform. "
                    "Describe your task concisely."
    ),
]

if os.getenv("IM_FEELING_LUCKY") == "true":
    print("Adding shell tool to agent tools")
    tools.append(Tool(
        name="BashCommandExecutor",
        func=bash_command_tool,
        description="Executes arbitrary bash expressions. "
                    "Be aware that commands like 'cd' will not persist across executions, "
                    "and 'rm', 'ln', and 'mv' are blocked. "
                    "Chain your commands to execute more complex tasks. You cannot use interactive commands."
                    "Do not modify any files using this tool."
                    "Never pipe into files on disk."
    ))

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
        agent_chain = ConversationalAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            verbose=True,
        )
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent_chain,
            tools=tools,
            memory=memory,
            verbose=True,
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
