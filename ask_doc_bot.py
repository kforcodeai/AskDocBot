import argparse
import logging
from src.config import ConfigManager
from src.qa_pipeline import QAPipeline
from src.utils.logger import setup_logger

logger = setup_logger()

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
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()