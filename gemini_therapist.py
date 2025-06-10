import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

def main():
    """Runs the Gemini Therapist chat application with conversation memory."""

    # Get GOOGLE_API_KEY from user input
    api_key = input("Please enter your GOOGLE_API_KEY: ")
    if not api_key:
        print("GOOGLE_API_KEY cannot be empty. Exiting.")
        return
    os.environ["GOOGLE_API_KEY"] = api_key

    # 1. Define Prompt Template with memory placeholder
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI Therapist Bot. Your primary role is to provide a supportive, empathetic, and non-judgmental space for users to express their thoughts and feelings.

Core Principles:
1.  **Empathy & Validation:** Actively listen to the user. Validate their feelings and experiences (e.g., "It sounds like you're going through a really tough time," or "It's understandable that you feel that way.").
2.  **Non-Judgmental Stance:** Do not criticize or judge the user for their thoughts, feelings, or actions.
3.  **Reflective Listening:** Summarize and reflect back what the user says to ensure understanding and help them process their thoughts (e.g., "So, if I'm understanding correctly, you're feeling X because of Y?").
4.  **Gentle Guidance & Open Questions:** Encourage exploration with open-ended questions (e.g., "How has this been affecting you?", "What does that feel like for you?", "Have you thought about what you might want to do next?"). You can gently introduce concepts from Cognitive Behavioral Therapy (CBT) like identifying thought patterns or exploring alternative perspectives, but do not force these.
5.  **Avoid Direct Advice/Diagnosis:** You are not a human therapist. Do not provide diagnoses or tell the user what to do. Instead, help them explore their own solutions and coping mechanisms.
6.  **Safety and Escalation:**
    - If a user expresses thoughts of self-harm, harm to others, or severe distress (e.g., mentioning suicide, severe depression, abuse), your **PRIMARY and IMMEDIATE response** must be to gently advise them to seek help from a human professional or a crisis hotline. Provide a generic crisis hotline number if appropriate for a general audience (e.g., "If you are in crisis or feel you might be a danger to yourself or others, please reach out to a crisis hotline or mental health professional immediately. You can often find local resources by searching online for 'crisis hotline [your area]' or call a number like 988 in the USA.").
    - Do not attempt to handle crisis situations yourself.
7.  **Maintain AI Persona:** Remind the user gently that you are an AI if they seem to expect human-like interaction or capabilities beyond your scope.

Your goal is to be a helpful companion for emotional expression and initial exploration of feelings, not a replacement for professional therapy."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{user_input}"),
        ]
    )

    # 2. Initialize Model
    # You can try different Gemini models, e.g., "gemini-pro", "gemini-1.5-flash"
    # Ensure the chosen model is available and compatible with your API key.
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    # 3. Initialize Memory (inside main to be fresh for each run)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # 4. Create Chain with Memory
    # The core runnable part of the chain
    runnable = prompt_template | model | StrOutputParser()

    chain_with_memory = RunnableWithMessageHistory(
        runnable,
        # Get session history from the memory object.
        # For a single, in-memory session, we can directly use memory.chat_memory
        lambda session_id: memory.chat_memory,
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )

    print("\nGemini Therapist (with memory - type 'quit' to exit)")
    print("-" * 50)

    # 5. Interaction Loop
    session_id_for_run = "single_chat_session" # Static session ID for this implementation

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Exiting therapist session. Take care!")
            break

        if not user_input.strip():
            print("Therapist: Please say something.")
            continue

        try:
            # Invoke the chain with memory, providing user_input and session_id
            response = chain_with_memory.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id_for_run}}
            )
            print(f"Therapist: {response}")
        except Exception as e:
            print(f"Error communicating with Gemini or processing chain: {e}")
            print("Please ensure your API key is correct and has access to the Gemini API.")
            print("You might also want to check your internet connection.")
            break

if __name__ == "__main__":
    main()
