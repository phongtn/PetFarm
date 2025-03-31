import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from utils import load_env
load_env()
github_key = os.getenv("GITHUB_TOKEN")
github_endpoint = os.getenv("GITHUB_ENDPOINT")


# Define a tool
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


async def main() -> None:
    primary_agent = AssistantAgent(
        name="primary_agent",
        model_client=leading_model(),
        # tools=[get_weather],
        system_message="You are a helpful assistant.",
        reflect_on_tool_use=True,
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=executing_model(),
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

    # When running inside a script, use an async main function and call it from `asyncio.run(...)`.
    # await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(task="Write a short poem about the fall season."):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message)


def leading_model():
    client = AzureAIChatCompletionClient(
        model='DeepSeek-R1',
        max_tokens=2048,
        endpoint='https://models.github.ai/inference',
        credential=AzureKeyCredential(github_key),
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
        }
    )
    return client


def executing_model():
    client = AzureAIChatCompletionClient(
        model='gpt-4o',
        endpoint='https://models.inference.ai.azure.com',
        credential=AzureKeyCredential(github_key),
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
        }
    )
    return client


if __name__ == '__main__':
    # asyncio.run(main())
    print(github_endpoint)
