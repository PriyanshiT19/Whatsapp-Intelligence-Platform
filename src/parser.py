# ============================================
# src/parser.py
# WhatsApp Chat Parser
# 
# Responsibility: Convert raw WhatsApp .txt
# file into a clean pandas DataFrame
#
# Input  → raw .txt file path
# Output → pandas DataFrame
# ============================================

import re
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────
TIMESTAMP_PATTERN = (
    r"(\d{1,2}/\d{1,2}/\d{2,4})"          # Group 1: Date
    r",\s"                                  # Comma + whitespace
    r"(\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?)"  # Group 2: Time (AM/PM optional)
    r"\s-\s"                                # Separator
    r"(.*)"                                 # Group 3: Rest of line
)

MEDIA_PATTERN   = "<Media omitted>"
DELETED_PATTERN = "This message was deleted"
GROUP_KEYWORDS  = ["joined", "left", "added", 
                   "removed", "changed", "created",
                   "invite link"]


# ── Helper Functions ───────────────────────────────────────────────────────

def is_new_message(line):
    """
    Check if a line is the start of a new message.
    Returns True if line starts with a timestamp pattern.
    
    Example:
        "3/14/23, 9:42 AM - Alice: Hi" → True
        "19/08/25, 13:23 - Alice: Hi"  → True
        "which continues here"          → False
    """
    return bool(re.match(TIMESTAMP_PATTERN, line))


def extract_parts(line):
    """
    Extract date, time, and content from a timestamped line.
    
    Returns:
        tuple: (date, time, content)
    
    Example:
        "3/14/23, 9:42 AM - Alice: Hi"
        → ("3/14/23", "9:42 AM", "Alice: Hi")
        
        "19/08/25, 13:23 - Alice: Hi"
        → ("19/08/25", "13:23", "Alice: Hi")
    """
    match = re.match(TIMESTAMP_PATTERN, line)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def split_sender_message(content):
    """
    Split "Sender: message" into (sender, message).
    Uses maxsplit=1 to handle colons inside messages.
    
    Returns:
        tuple: (sender, message)
    
    Examples:
        "Alice: Hello!"              → ("Alice", "Hello!")
        "Alice: https://google.com"  → ("Alice", "https://google.com")
        "Dave joined the group"      → ("System", "Dave joined the group")
    """
    parts = content.split(":", 1)
    
    if len(parts) == 2:
        sender  = parts[0].strip()
        message = parts[1].strip()
    else:
        sender  = "System"
        message = content.strip()
    
    return sender, message


def get_message_type(sender, message):
    """
    Classify a message into its type.
    
    Types:
        normal      → regular text message
        media       → image/video/audio shared
        deleted     → message was deleted by sender
        system      → WhatsApp system notification
        group_event → someone joined/left/was added
    
    Returns:
        str: message type
    """
    if sender == "System":
        if any(kw in message.lower() for kw in GROUP_KEYWORDS):
            return "group_event"
        return "system"
    
    if MEDIA_PATTERN in message:
        return "media"
    
    if DELETED_PATTERN in message:
        return "deleted"
    
    return "normal"


# ── Main Parser Function ───────────────────────────────────────────────────

def parse_whatsapp_chat(file_path):
    """
    Parse a WhatsApp exported chat .txt file.
    
    Parameters:
        file_path (str): Path to WhatsApp .txt export
    
    Returns:
        pd.DataFrame with columns:
            date         → string date
            time         → string time
            sender       → person who sent the message
            message      → full message text
            message_type → normal/media/deleted/system/group_event
    
    Raises:
        FileNotFoundError: if file path is wrong
        ValueError: if file has no parseable messages
    """
    
    # ── Step 1: Read file ──────────────────────────────
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Chat file not found at: {file_path}\n"
            f"Please check the file path."
        )
    
    # ── Step 2: Parse line by line ─────────────────────
    messages = []
    
    current_date    = None
    current_time    = None
    current_sender  = None
    current_message = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        if is_new_message(line):
            
            # Save previous message
            if current_message is not None:
                messages.append({
                    "date"        : current_date,
                    "time"        : current_time,
                    "sender"      : current_sender,
                    "message"     : current_message,
                    "message_type": get_message_type(
                                        current_sender,
                                        current_message
                                    )
                })
            
            # Start new message
            current_date, current_time, content = extract_parts(line)
            current_sender, current_message     = split_sender_message(content)
        
        else:
            # Continuation line
            if current_message is not None:
                current_message = current_message + " " + line
    
    # Save the last message
    if current_message is not None:
        messages.append({
            "date"        : current_date,
            "time"        : current_time,
            "sender"      : current_sender,
            "message"     : current_message,
            "message_type": get_message_type(
                                current_sender,
                                current_message
                            )
        })
    
    # ── Step 3: Validate output ────────────────────────
    if len(messages) == 0:
        raise ValueError(
            "No messages found. Check if file format matches "
            "expected WhatsApp export format."
        )
    
    # ── Step 4: Create DataFrame ───────────────────────
    df = pd.DataFrame(messages)
    
    return df


# ── Quick Test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "sample", "sample_chat.txt"
    )
    
    print("Testing parser with sample file...")
    df = parse_whatsapp_chat(sample_path)
    
    print(f"✅ Messages parsed : {len(df)}")
    print(f"✅ Columns         : {list(df.columns)}")
    print(f"✅ Senders         : {df['sender'].unique()}")
    print(f"\nMessage types:")
    print(df['message_type'].value_counts())
    print(f"\nFirst 3 rows:")
    print(df.head(3))