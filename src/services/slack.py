from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging

logger = logging.getLogger(__name__)

class SlackService:
    """Service class to encapsulate Slack integration."""

    def __init__(self, slack_token: str):
        self.client = WebClient(token=slack_token)

    def post_message(self, channel: str, message: str) -> None:
        """Post a message to a Slack channel."""
        try:
            self.client.chat_postMessage(channel=channel, text=message)
            logger.info(f"Message posted to {channel}.")
        except SlackApiError as e:
            logger.error(f"Error posting to Slack: {e.response['error']}")