# AskDocBot

AskDocBot is an intelligent document question-answering cli tool leveraging openai models, designed to parse a PDF file, answer a set of questions, and, upon request, post results directly to a Slack channel. This repository supports both direct and Slack-based Q&A on PDF documents, providing structured JSON responses.

## Table of Contents

- [Features](#features)
- [Supported Input Types](#supported-input-types)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Example Command](#example-command)
- [Output Format](#output-format)
- [Slack Integration](#slack-integration)
- [Contributing](#contributing)
- [License](#license)

## Features

- **PDF-based Question Answering**: Extract answers from a given PDF based on provided questions.
- **Semantic Chunking**: Uses spacy `en_core_web_sm` model to provide semantic chunking by chunking at sections, headers and max_tokens
- **Similarity Search**: Can be used with any sentence-transformer model, just have to change the `embedding_model_name` and `embedding_dimension` varibales in config, by default leverages `infgrad/stella-base-en-v2` which is just a `220 mb` and ranks 54 on MTEB as of `Nov 1 2024`
- **Confidence-based Responses**: Low-confidence answers are flagged as “Data Not Available.”
- **Leverages concurrency**: Wherever possible leverages multithreads to make the processing a bit faster
- **Slack Posting**: Results can be posted to a specified Slack channel if prompted.
- **JSON Output**: Results are structured in JSON format for easy parsing.

## Supported Input Types

- **PDF Files**: Input documents should be in PDF format.
- **Question List**: Questions are provided as a list of strings.

## Setup

### Prerequisites

- **Python 3.11+**
- **Slack API Token**: Required for posting results to Slack. Not Mandatory
- **OpenAI API Key**: Required for processing document-based Q&A.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory and add the following keys:

```plaintext
OPENAI_API_KEY=<Your OpenAI API Key>
SLACK_API_TOKEN=<Your Slack API Token>
```

## Usage

### Running the Application

To start the application, pass the PDF path, list of questions, and user command. Use the following format:

```python
python ask_doc_bot.py
```

### Example Command

This example processes a PDF with specified questions and posts results to Slack if requested.

```python
# In your Python file or interactive shell
pdf_path = "path/to/your/document.pdf"
questions = [
    "What is the company name?",
    "Who is the CEO of the company?",
    "What is the vacation policy?",
    "What is the termination policy?"
]
user_command = "Answer the questions and post results on Slack."
slack_channel = "#your-channel"

main(pdf_path, questions, user_command, slack_channel)
```

## Output Format

The output is structured as a JSON file, with each question-answer pair formatted as follows:

```json
{
  "results": [
    {
      "question": "What is the company name?",
      "answer": "Zania, Inc.",
      "confidence_score": 0.95,
      "source_pages": [3]
    },
    {
      "question": "What is the vacation policy?",
      "answer": "Data Not Available",
      "confidence_score": 0.45,
      "source_pages": []
    }
  ],
  "timestamp": "YYYY-MM-DD_HH:MM:SS"
}
```

The JSON file is saved in the specified `output_folder` and may be posted to Slack as a structured message.

## Slack Integration

The application can post results to a Slack channel if requested. To set up Slack integration:

1. [Create a Slack App](https://api.slack.com/apps) with permissions for posting.
2. Install the app in your workspace and obtain the OAuth token.
3. Set the token as `SLACK_API_TOKEN` in your `.env` file.
4. Specify the Slack channel in the `main` function when running the app.

## For quick demo run `bash run.sh`