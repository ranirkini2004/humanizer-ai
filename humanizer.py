from transformers import pipeline

# Load the model globally so it only initializes once on startup.
# We use flan-t5-small for memory efficiency on free hosting tiers (Render/Railway).
# You can easily change this to "google/flan-t5-base" or "google/flan-t5-large" for better quality.
MODEL_NAME = "google/flan-t5-small"

print(f"Loading AI Model: {MODEL_NAME}... This may take a minute.")
# text2text-generation is the correct pipeline for T5 models
humanizer_pipeline = pipeline("text2text-generation", model=MODEL_NAME)
print("Model loaded successfully!")


def humanize_text(text: str, style: str = "Casual") -> str:
    """
    Takes AI-generated text and rewrites it in a specific human-like style.
    """
    # Construct an instruction prompt for the FLAN-T5 model
    prompt = f"Rewrite the following text to sound like a human wrote it in a {style} tone: {text}"

    # Generate the text
    # max_length: limits output size, do_sample & temperature: adds natural randomness
    result = humanizer_pipeline(
        prompt,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return result[0]['generated_text']