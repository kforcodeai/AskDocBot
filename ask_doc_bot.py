import os
import json
import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from configparser import ConfigParser
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.openai_qa import OpenAIPDFQuestionAnswering
from src.data_models import QAEncoder
import spacy
import sys
import subprocess

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_and_download_spacy_model():
    """Check if spaCy model is installed; if not, download it."""
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        logger.info("SpaCy model 'en_core_web_sm' not found. Downloading...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )

def load_config():
    """Load environment variables and configuration file."""
    load_dotenv(find_dotenv())
    config = ConfigParser()
    config.read("config.ini")
    return config

def get_api_keys(config):
    """Retrieve API keys from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    slack_token = os.getenv("SLACK_API_TOKEN")
    if not api_key or not slack_token:
        logger.error("API_KEY or SLACK_API_TOKEN is missing in environment variables.")
        raise EnvironmentError("API_KEY or SLACK_API_TOKEN is missing in environment variables.")
    return api_key, slack_token

def initialize_artifacts_folder(config):
    """Ensure that the artifacts folder exists."""
    artifacts_folder = config.get("DEV", "output_folder", fallback="artifacts")
    os.makedirs(artifacts_folder, exist_ok=True)
    return artifacts_folder

def post_to_slack(slack_token, channel, message):
    """Post a message to a specified Slack channel."""
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        return response
    except SlackApiError as e:
        logger.error(f"Error posting to Slack: {e.response['error']}")

def parse_user_input(user_command):
    """Parse the user's command to determine if results should be posted to Slack."""
    return "post results on Slack" in user_command.lower()

def format_results_json(questions, answers):
    """Structure the answers into the desired JSON format."""
    results = {"results": []}
    for qa_response in answers.get("results", []):
        question = qa_response.question
        answer = qa_response.answer
        confidence = qa_response.confidence_score
        source_pages = qa_response.source_pages
        if confidence < 0.5:
            answer = "Data Not Available"
        results["results"].append({
            "question": question,
            "answer": answer,
            "confidence_score": confidence,
            "source_pages": source_pages,
        })
    return results

def main(pdf_path, questions, user_command, slack_channel):
    config = load_config()
    api_key, slack_token = get_api_keys(config)
    artifacts_folder = initialize_artifacts_folder(config)
    check_and_download_spacy_model()
    
    # Pass logger to the OpenAIPDFQuestionAnswering instance
    qa_system = OpenAIPDFQuestionAnswering(config, api_key, logger)
    answers = qa_system.process_questions(pdf_path, questions)
    results = format_results_json(questions, answers)
    
    output_file = os.path.join(artifacts_folder, f"qaresults_{answers['timestamp']}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, cls=QAEncoder, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    if parse_user_input(user_command):
        slack_message = json.dumps(results, indent=2)
        post_to_slack(slack_token, slack_channel, slack_message)
        logger.info(f"Results posted to Slack channel: {slack_channel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for PDF-based question answering.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document.")
    parser.add_argument("questions", nargs="+", type=str, help="List of questions to answer.")
    parser.add_argument("--user_command", type=str, default="", help="Command with instructions for posting results.")
    parser.add_argument("--slack_channel", type=str, default="#general", help="Optional Slack channel to post results.")
    args = parser.parse_args()
    main(args.pdf_path, args.questions, args.user_command, args.slack_channel)