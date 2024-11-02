import argparse
from src.qa_pipeline import QAPipeline
from configparser import ConfigParser
from dotenv import load_dotenv, find_dotenv
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    openai_api_key: str
    slack_token: str

class ConfigManager:
    def __init__(self):
        self._load_env()
        self.config = self._load_config()
        self.api_config = self._load_api_config()

    def _load_env(self):
        load_dotenv(find_dotenv())

    def _load_config(self) -> ConfigParser:
        config = ConfigParser()
        config.read("config.ini")
        return config

    def _load_api_config(self) -> APIConfig:
        api_key = os.getenv("OPENAI_API_KEY")
        slack_token = os.getenv("SLACK_API_TOKEN")
        
        if not api_key or not slack_token:
            raise EnvironmentError(
                "OPENAI_API_KEY or SLACK_API_TOKEN is missing in environment variables."
            )
        
        return APIConfig(openai_api_key=api_key, slack_token=slack_token)

    def get_config(self) -> ConfigParser:
        return self.config

    def get_api_config(self) -> APIConfig:
        return self.api_config

def parse_arguments():
    parser = argparse.ArgumentParser(description="CLI for PDF-based question answering.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document.")
    parser.add_argument("questions", nargs="+", type=str, help="List of questions to answer.")
    parser.add_argument(
        "--user_command",
        type=str,
        default="",
        help="Command with instructions for posting results."
    )
    parser.add_argument(
        "--slack_channel",
        type=str,
        default="#general",
        help="Optional Slack channel to post results."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    config_manager = ConfigManager()
    
    try:
        pipeline = QAPipeline(config_manager)
        pipeline.process(
            pdf_path=args.pdf_path,
            questions=args.questions,
            user_command=args.user_command,
            slack_channel=args.slack_channel
        )
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()