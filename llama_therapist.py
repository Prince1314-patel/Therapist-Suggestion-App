import os
import torch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

def main():
    """Runs the Llama Therapist chat application with conversation memory."""

    # --- 1. Model Loading (Local Pipeline) ---
    # Note: Ensure you have access to this model and have accepted its terms on Hugging Face.
    # You might need to log in via `huggingface-cli login` if you haven't already.
    # If "meta-llama/Llama-2-7b-chat-hf" is too large or you don't have access,
    # consider smaller models like "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # or other open-source chat models you have downloaded.
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Smaller alternative

    try:
        print(f"Loading tokenizer for {model_id}...")
        # It's good practice to load the tokenizer separately to handle potential trust issues or specific configurations.
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Add pad_token if missing (common for some Llama versions)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Creating Hugging Face text-generation pipeline for {model_id}...")
        # Determine torch_dtype based on GPU availability
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            print("GPU detected. Using torch.float16 for model.")
        else:
            # Using bfloat16 if CUDA is not available but CPU supports it (common on modern CPUs)
            # Fallback to float32 if not, though this might be very slow for large models.
            # Transformers pipeline usually handles this well by default if not specified.
            torch_dtype = torch.bfloat16 if torch.backends.mps.is_available() or hasattr(torch.ops.aten, 'cpu_supports_bf16') else torch.float32
            print(f"No GPU detected or not configured for Transformers. Using {torch_dtype} for model. This might be slow.")

        hf_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device_map="auto",  # Automatically uses GPU if available
            max_new_tokens=512,  # Max tokens to generate in the response
            # You can add other pipeline arguments here, e.g.:
            # temperature=0.7,
            # top_k=50,
            # do_sample=True,
            pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id
        )
        print("Pipeline created successfully.")

        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("HuggingFacePipeline initialized.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure all necessary libraries (transformers, torch, accelerate) are installed correctly.")
        print("You might need to run: pip install transformers torch accelerate bitsandbytes sentencepiece")
        return
    except OSError as e:
        print(f"OSError: {e}. This often means the model ID '{model_id}' is incorrect, you don't have access, or the model files are corrupted.")
        print("Please double-check the model_id. If it's a gated model (like Llama 2), ensure you have access and are logged in via `huggingface-cli login`.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model loading or pipeline creation: {e}")
        print("Check your internet connection, model access rights, and available disk space/memory.")
        return

    # --- 2. Define Prompt Template with memory placeholder ---
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an AI Therapist Bot (Llama version). Your primary role is to provide a supportive, empathetic, and non-judgmental space for users to express their thoughts and feelings.

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
8.  **Conciseness:** Keep your responses supportive yet reasonably concise, especially for this Llama version.

Your goal is to be a helpful companion for emotional expression and initial exploration of feelings, not a replacement for professional therapy."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{user_input}"),
        ]
    )

    # --- 3. Initialize Memory (after model loading, before chain creation) ---
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # --- 4. Create Chain with Memory ---
    runnable = prompt_template | llm | StrOutputParser()

    chain_with_memory = RunnableWithMessageHistory(
        runnable,
        lambda session_id: memory.chat_memory, # Get session history from the memory object
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )

    print("\nLlama Therapist (with memory - type 'quit' to exit)")
    print("-" * 50)

    # --- 5. Interaction Loop ---
    session_id_for_run = "llama_chat_session" # Static session ID for this implementation

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Exiting therapist session. Take care!")
            break

        if not user_input.strip():
            print("Therapist: Please say something.")
            continue

        try:
            response = chain_with_memory.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id_for_run}}
            )
            print(f"Therapist: {response}")
        except Exception as e:
            print(f"Error during model inference or chain processing: {e}")
            # Don't break the loop for inference errors, allow user to try again or quit.
            print("There was an issue getting a response. Please try again.")

if __name__ == "__main__":
    # It's good practice to set HUGGING_FACE_HUB_TOKEN if needed,
    # though for public models or already logged-in users, it might not be strictly necessary.
    # if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    # print("Note: HUGGING_FACE_HUB_TOKEN environment variable not set. This might be required for private/gated models.")
    main()
