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