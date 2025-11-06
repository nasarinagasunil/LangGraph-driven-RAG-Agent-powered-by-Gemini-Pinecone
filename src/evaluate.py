import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

judge = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))

def evaluate(question, answer):
    prompt = f"""
You are an evaluator model.

Question: {question}

Model Answer: {answer}

Evaluate the answer: Is the answer correct and relevant to the question?

Reply ONLY in this format:
VERDICT: YES or NO
REASON: short one sentence reason
"""
    result = judge.invoke(prompt)
    return result

if __name__ == "__main__":
    print("\nEnter Question:")
    q = input("> ")

    print("\nEnter Model Answer:")
    a = input("> ")

    print("\n===== EVALUATION RESULT =====")
    print(evaluate(q, a))
