import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print("=" * 60)
    print("Testing Groq API / Token Quota")
    print("=" * 60)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: GROQ_TEST_OK",
            }
        ],
        temperature=0,
        max_tokens=20,
    )

    answer = response.choices[0].message.content

    print("\nModel response:")
    print(answer)

    print("\nUsage:")
    print(response.usage)

    print("\n" + "=" * 60)
    print("Groq API test completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()