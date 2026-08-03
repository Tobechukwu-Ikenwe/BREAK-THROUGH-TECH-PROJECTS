# === Imports ===

import os                            # For interacting with the operating system (file paths, folders)
import glob                          # For finding files matching a specific pattern
import warnings                      # To manage warning messages (we'll suppress some syntax ones)
import markdown                      # To convert Markdown text to HTML
import ffmpeg                        # For audio manipulation (splitting MP3s)
from openai import OpenAI
from IPython.display import display, HTML, Markdown as IPMarkdown

# === OpenAI Client Setup ===
client = OpenAI()

def split_audio_file(input_file: str, output_folder: str, segment_length_minutes: int = 10) -> bool:
    """
    Splits an audio file into segments of specified duration.
    
    Args:
        input_file: Path to the source MP3 file
        output_folder: Directory to save the segments
        segment_length_minutes: Length of each segment in minutes
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n--- Starting Audio Splitting ---")
    print(f"Input: {input_file}")
    print(f"Output folder: {output_folder}")
    print(f"Segment length: {segment_length_minutes} minutes")
    
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return False
    
    try:
        print("Loading audio...")
    
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
    
        segment_length_seconds = segment_length_minutes * 60
        output_pattern = os.path.join(output_folder, "segment_%03d.mp3")
    
        print("Splitting audio with ffmpeg...")
    
        (
            ffmpeg
            .input(input_file)
            .output(
                output_pattern,
                f="segment",
                segment_time=segment_length_seconds,
                c="copy"
            )
            .run(quiet=True)
        )
    
        # Count created segments
        segment_count = len([
            f for f in os.listdir(output_folder) #for each file in output folder
            if f.startswith("segment_") and f.endswith(".mp3")
        ])
    
        print(f"\n✓ Created {segment_count} segments")
        return True

    except Exception as e:
        print(f"Error during splitting: {e}")
        return False

  # Input audio file
lecture_audio_file = "data/audio_input/lecture-04.mp3"

# Intermediate storage
segments_folder = "data/segments"
transcripts_folder = "data/transcriptions"

# Output
combined_transcript_file = os.path.join(transcripts_folder, "combined_transcript.txt")
notes_output_folder = "data/lecture_notes"

# Parameters
segment_length_min = 10
ai_model = "gpt-4o"

split_successful = split_audio_file(
    input_file=lecture_audio_file,
    output_folder=segments_folder,
    segment_length_minutes=segment_length_min
)

if split_successful:
    print("\n✓ Audio splitting completed successfully!")
else:
    print("\n✗ Audio splitting encountered errors.")

def transcribe_audio_segment(audio_path: str, output_path: str) -> bool:
    """
    Transcribes a single audio segment using Whisper API.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the transcript
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Transcribing: {os.path.basename(audio_path)}")
    
    try:
        # Open the audio file in binary mode
        with open(audio_path, "rb") as audio_file:
            # Send to Whisper API for transcription
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Extract the transcribed text
        text = transcription.text
        
        # Save transcript to file
        with open(output_path, "w", encoding="utf-8") as f:   #w for write
            f.write(text)
        
        print(f"  ✓ Saved to {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

print("Transcription function defined successfully!")


print("\n--- Transcribing Audio Segments ---\n")

# Get all segment files
segment_files = sorted([f for f in os.listdir(segments_folder) if f.endswith(".mp3")])

if not segment_files:
    print("Error: No segment files found!")
else:
    print(f"Found {len(segment_files)} segments to transcribe\n")
    
    # Create transcripts folder
    os.makedirs(transcripts_folder, exist_ok=True)
    
    # Transcribe each segment
    success_count = 0
    for segment_file in segment_files:
        audio_path = os.path.join(segments_folder, segment_file)
        transcript_file = segment_file.replace(".mp3", ".txt")
        transcript_path = os.path.join(transcripts_folder, transcript_file)
        
        if transcribe_audio_segment(audio_path, transcript_path):
            success_count += 1
    
    # YOUR CODE HERE
    print(success_count)
    # END OF YOUR CODE


def combine_transcripts(folder: str, output_file: str) -> str:
    """
    Combines multiple transcript files into one.
    
    Args:
        folder: Directory containing transcript .txt files
        output_file: Path to save the combined transcript
    
    Returns:
        The combined transcript text
    """
    print("\n--- Combining Transcripts ---\n")
    
    # Find all transcript files and sort them
    transcript_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    
    if not transcript_files:
        print("Error: No transcript files found!")
        return None
    
    print(f"Found {len(transcript_files)} transcripts to combine")
    
    # Read and combine
    all_content = []
    for file_path in transcript_files:
        with open(file_path, 'r', encoding="utf-8") as f:
            all_content.append(f.read())
    
    combined = "\n".join(all_content)
    
    # Save combined transcript
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"✓ Combined transcript saved to {output_file}")
    print(f"  Total length: {len(combined):,} characters")
    
    return combined

#combint the transcripts 

full_transcript = combine_transcripts(transcripts_folder, combined_transcript_file)

if full_transcript:
    print("\n--- Preview of Combined Transcript ---")
    print(full_transcript[:500] + "...")


def generate_notes(transcript: str, system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> str:
    """
    Generates notes from a transcript using AI.
    
    Args:
        transcript: The full transcript text
        system_prompt: System-level instructions
        user_prompt: Specific task instructions
        model: The AI model to use
    
    Returns:
        Generated notes in Markdown format
    """
    print(f"\n--- Generating Notes (using {model}) ---\n")
    
    if not transcript:
        print("Error: No transcript provided!")
        return None
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nTranscript:\n{transcript}"}
        ]
        
        print("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        notes = response.choices[0].message.content
        print("✓ Notes generated successfully")
        return notes
        
    except Exception as e:
        print(f"Error generating notes: {e}")
        return None


if simple_notes:
    print("--- Simple Prompt Results ---\n")
    display(IPMarkdown(simple_notes[:2000] + "\n\n...(truncated for preview)..."))


detailed_system = """You are an expert academic assistant specializing in creating 
comprehensive, pedagogically sound lecture notes."""

detailed_user = """Please generate comprehensive lecture notes from the following transcript.

Incorporate these advanced learning techniques:

1. **Structure**: Use clear hierarchical organization (main topics → subtopics → key points)
2. **Summary**: Begin with learning objectives or key takeaways
3. **Formatting**: Use bullet points, numbered lists, and clear headings
4. **Emphasis**: Highlight key terms, definitions, and important concepts using **bold** or *italics*
5. **Examples**: Include relevant examples and real-world applications
6. **Engagement**: Add thought-provoking questions or discussion points
7. **Analogies**: Provide analogies or metaphors for difficult concepts
8. **Self-Check**: Insert brief "check your understanding" sections
9. **Memory Aids**: Include mnemonics where applicable
10. **Connections**: Add cross-references to related concepts
11. **Conclusion**: End with main takeaways and areas for further exploration

Feel free to expand on the content with relevant background information and fill in gaps 
to ensure completeness. The goal is comprehensive, engaging notes that facilitate deep 
understanding and retention.
"""

detailed_notes = generate_notes(full_transcript, detailed_system, detailed_user, ai_model)

if detailed_notes:
    print("--- Detailed Prompt Results ---\n")
    display(IPMarkdown(detailed_notes[:2000] + "\n\n...(truncated for preview)..."))


def save_and_display_notes(notes: str, output_dir: str, filename: str = "lecture_notes"):
    """
    Saves notes as Markdown and HTML, and displays them.
    
    Args:
        notes: The notes content in Markdown format
        output_dir: Directory to save the files
        filename: Base filename (without extension)
    """
    if not notes:
        print("No notes to save.")
        return
    
    print(f"\n--- Saving Notes ---\n")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Markdown
    md_path = os.path.join(output_dir, f"{filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"✓ Markdown saved: {md_path}")
    
    # Convert to HTML and save
    html_content = markdown.markdown(notes)
    html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{filename}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; margin-top: 1.5em; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 20px; color: #666; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    html_path = os.path.join(output_dir, f"{filename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"✓ HTML saved: {html_path}")
    
    # Display in notebook
    print("\n--- Preview of Generated Notes ---\n")
    display(HTML(html_content))




def save_and_display_notes(notes: str, output_dir: str, filename: str = "lecture_notes"):
    """
    Saves notes as Markdown and HTML, and displays them.
    
    Args:
        notes: The notes content in Markdown format
        output_dir: Directory to save the files
        filename: Base filename (without extension)
    """
    if not notes:
        print("No notes to save.")
        return
    
    print(f"\n--- Saving Notes ---\n")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Markdown
    md_path = os.path.join(output_dir, f"{filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"✓ Markdown saved: {md_path}")
    
    # Convert to HTML and save
    html_content = markdown.markdown(notes)
    html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{filename}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; margin-top: 1.5em; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 20px; color: #666; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    html_path = os.path.join(output_dir, f"{filename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"✓ HTML saved: {html_path}")
    
    # Display in notebook
    print("\n--- Preview of Generated Notes ---\n")
    display(HTML(html_content))








"""

FULL PIPELINE
# 1. Split audio into manageable segments
split_audio_file(lecture_audio_file, segments_folder, segment_length_min)

# 2. Transcribe each segment with Whisper
for each segment:
    transcribe_audio_segment(segment, output)

# 3. Combine all transcripts into one
full_transcript = combine_transcripts(transcripts_folder, combined_file)

# 4. Generate comprehensive notes with AI
notes = generate_notes(full_transcript, system_prompt, user_prompt)

# 5. Save and display the results
save_and_display_notes(notes, output_folder)

"""
