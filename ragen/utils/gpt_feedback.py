import os
from openai import OpenAI

client = None

def get_fuzzy_feedback(question, student_answer, model="gpt-4o"):
    global client
    # Only enable if explicitly requested
    if os.environ.get("ENABLE_GPT_FEEDBACK", "0") != "1":
        return "Incorrect. Please think again."

    print(f"[DEBUG] Requesting GPT feedback for answer length {len(student_answer)}")

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[Warning] ENABLE_GPT_FEEDBACK=1 but OPENAI_API_KEY is not set.")
            return "Incorrect. Please think again."
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
             print(f"[ERROR] Failed to init OpenAI client: {e}")
             return "Incorrect. Please think again."

    sys_prompt = "You are a strict but friendly math TA. Imitate a real human and provide ambiguous or even slightly inconsistent feedback that nudges directionally, without revealing the final answer or any specific numbers. Use 2-3 sentences."
    user_prompt = f"Problem:\n{question}\n\nStudent answer:\n{student_answer}\n\nPlease provide ambiguous, non-revealing feedback (no final answer, no specific numbers)."

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=180,
        )
        feedback = resp.choices[0].message.content
        print(f"[GPT Feedback] Generated: {feedback[:50]}...")
        return feedback
    except Exception as e:
        print(f"[Error] GPT feedback failed: {e}")
        return "Incorrect. Please think again."

