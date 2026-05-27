import dotenv
import os
import asyncio
from github import Github
from github import GithubException
from github import Auth
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, AgentOutput, ToolCall, ToolCallResult, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate

# == Initializations ==
dotenv.load_dotenv()

# Model
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# Github
git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN"))) if os.getenv("GITHUB_TOKEN") else None
full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")


# == Functions ==
def get_pr_details(pull_number:int) -> dict:
    """
    Provides details about a pull request (pr) given a pull request number
    :param pull_number: pull request number
    :return: dictionary containing details of the pull request
    """
    pr_details = {}
    try:
        repo = git.get_repo(full_repo_name)
        pull = repo.get_pull(pull_number)
        pr_details['author'] = pull.user.login
        pr_details['title'] = pull.title
        pr_details['body'] = pull.body
        pr_details['state'] = pull.state
        pr_details['diff_url'] = pull.diff_url
        pr_details['head_sha'] = pull.head.sha
        return pr_details
    except GithubException as e:
        raise ValueError(f"An error occurred while accessing the repository: {e.data.get('message', 'No error message')}")

def get_file_contents(path:str) -> str:
    """
    Provides contents of a file given a path
    :param path: Path of the file within the repository
    :return: File contents
    """
    try:
        repo = git.get_repo(full_repo_name)
        file_content = repo.get_contents(path).decoded_content.decode('utf-8')
        return file_content
    except GithubException as e:
        return "{'error': 'Unable to retrieve file contents'}"

def get_pr_commit_details(commit_sha:str) -> list:
    """
    Provides details about a pull request (pr) given a commit sha
    :param commit_sha: SHA of the commit to retrieve details for
    :return: list containing details of the pull request
    """
    try:
        repo = git.get_repo(full_repo_name)
        commit = repo.get_commit(commit_sha)
        changed_files: list[dict] = []
        # for f in commit.files:
        for f in commit.get_files():
            changed_files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch
            })
        return changed_files
    except GithubException as e:
        return [{'error': 'Unable to retrieve commit details'}]

async def add_context_to_state(context_summary:str):
    """
    Adds gathered context to the state
    :param context_summary: Context summary after gathering the context
    :return: None
    """
    async with context.store.edit_state() as state:
        state["context_summary"] = context_summary

async def add_comment_to_state(draft_comment:str):
    """
    Adds review comments to the state
    :param draft_comment: Review comment
    :return: None
    """
    async with context.store.edit_state() as state:
        state["draft_comment"] = draft_comment

async def add_review_to_state(final_review:str):
    """
    Adds final review to the state
    :param final_review: Final and reviewed comment
    :return: None
    """
    async with context.store.edit_state() as state:
        state["final_review"] = final_review

def post_review_to_github(pr_number: int, comment:str) -> list:
    """
    Takes a final comment and posts it to the GitHub API
    :param pr_number: pull request number to add the comment to
    :param comment: final comment to add to the pull request
    :return:
    """
    try:
        repo = git.get_repo(full_repo_name)
        review = repo.get_pull(pr_number).create_review(body=comment)
        return [{'success': f'Review posted successfully with state {review.state}'}]
    except GithubException as e:
        return [{'error': {e.data.get('message')} }]


# == Tool ==
tools = [
    FunctionTool.from_defaults(get_pr_details),
    FunctionTool.from_defaults(get_file_contents),
    FunctionTool.from_defaults(get_pr_commit_details),
    FunctionTool.from_defaults(add_comment_to_state),
    FunctionTool.from_defaults(add_context_to_state),
    FunctionTool.from_defaults(add_review_to_state)
]

# == Agents ==
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context by commentor agent to draft pull requests comments.",
    tools=[FunctionTool.from_defaults(get_pr_details),
           FunctionTool.from_defaults(get_pr_commit_details),
           FunctionTool.from_defaults(add_context_to_state),
           FunctionTool.from_defaults(get_file_contents)],
    can_handoff_to=["CommentorAgent"],
    system_prompt="""
        You are the context agent. You MUST call the get_pr_details tool providing the PR number to gather: \n: 
      - The PR details: author, title, body, diff_url, state, and head_sha; \n
      - Changed files; \n
      - Any requested for files; \n
        Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[FunctionTool.from_defaults(add_comment_to_state)],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    system_prompt="""
        You are the commentor agent in charge of drafting a comment of a PR as a human reviewer would. \n 
        You MUST call the ContextAgent to request: \n
         - the PR details, 
         - commit details to obtain the changed files, 
         - file contents and 
         - any other repo files you may need, and wait for the results.
        IMPORTANT! Do not handoff to other agent until calling the ContextAgent for the PR details.
        After obtaining the PR context from the ContextAgent, ensure to do the following for a thorough review: 
         - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
            - What is good about the PR? \n
            - Did the author follow ALL contribution rules? What is missing? \n
            - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
            - Are new endpoints documented? - use the diff to determine this. \n 
            - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
         - You should directly address the author. So your comments should sound like: \n
            "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
        Once you have enough information about the PR, after calling the ContextAgent at least twice, you must hand off to the ReviewAndPostingAgent. 
    """
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Checks the draft pull review and if valid then it posts it to GitHub.",
    tools=[FunctionTool.from_defaults(add_review_to_state),
           FunctionTool.from_defaults(post_review_to_github)],
    can_handoff_to=["CommentorAgent"],
    system_prompt="""
        You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
        Once a review is generated, you need to run a final check and post it to GitHub.
           - The review must: \n
           - Be a ~200-300 word review in markdown format. \n
           - Specify what is good about the PR: \n
           - Did the author follow ALL contribution rules? What is missing? \n
           - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
           - Are there notes on whether new endpoints were documented? \n
           - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
         If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
         When you are satisfied, post the review to GitHub.  
    """
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": ""
    }
)

# == Context ==
context = Context(workflow_agent)

# == Execution ==
async def main():
    query = f"Write a review for PR number {pr_number}."
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()

