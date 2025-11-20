import os
from openai import OpenAI

client = None

def get_fuzzy_feedback(question, student_answer, model="gpt-4o"):
    global client
    # Only enable if explicitly requested
    if os.environ.get("ENABLE_GPT_FEEDBACK", "0") != "1":
        return "Incorrect. Please think again."

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Incorrect. Please think again."
        try:
            client = OpenAI(api_key=api_key)
        except:
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
        return feedback
    except:
        return "Incorrect. Please think again."

