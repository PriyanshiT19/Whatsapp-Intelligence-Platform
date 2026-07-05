# ============================================
# src/preprocessor.py
# WhatsApp Chat Preprocessor
#
# Responsibility: Take raw parsed DataFrame
# and add analytical features
#
# Input  → raw DataFrame from parser.py
# Output → enriched DataFrame with features
# ============================================

import re
import emoji
import pandas as pd


# ── Constants ──────────────────────────────────────────

URL_PATTERN = (
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
    r'|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)

DATETIME_FORMATS = [
    "%m/%d/%y %I:%M %p",    # 3/14/23 9:42 AM
    "%m/%d/%Y %I:%M %p",    # 3/14/2023 9:42 AM
    "%d/%m/%y %I:%M %p",    # 14/3/23 9:42 AM
    "%d/%m/%Y %I:%M %p",    # 14/3/2023 9:42 AM
]


# ── Helper Functions ───────────────────────────────────

def count_emojis(text):
    """
    Count number of emojis in a text string.
    
    Example:
        "Hello 😊 World 🎉" → 2
    """
    return len([c for c in str(text) 
                if c in emoji.EMOJI_DATA])


def count_words(text):
    """
    Count words in text.
    Splits on whitespace.
    
    Example:
        "Hello World" → 2
    """
    return len(str(text).split())


def count_urls(text):
    """
    Count URLs in text.
    
    Example:
        "Check https://google.com please" → 1
    """
    return len(re.findall(URL_PATTERN, str(text)))


def parse_datetime(datetime_str):
    """
    Try multiple datetime formats.
    Returns parsed datetime or None if all fail.
    
    WHY: Different phones export different formats.
    We try each format until one works.
    """
    for fmt in DATETIME_FORMATS:
        try:
            return pd.to_datetime(datetime_str, format=fmt)
        except:
            continue
    return None


# ── Main Preprocessing Function ────────────────────────

def preprocess(df):
    """
    Add analytical features to parsed DataFrame.
    
    Takes raw DataFrame from parser.py and adds:
    - Proper datetime column
    - Year, month, day, hour, weekday
    - Message length, word count
    - Emoji count, URL count
    - Boolean flags for media, deleted, URLs
    
    Parameters:
        df (pd.DataFrame): Raw parsed DataFrame
    
    Returns:
        pd.DataFrame: Enriched DataFrame with all features
    """
    
    # Make a copy — never modify original data
    # This is a critical Data Science best practice
    df = df.copy()
    
    # ── Step 1: Parse datetime ─────────────────────────
    df['datetime_str'] = df['date'] + " " + df['time']
    
    # Try parsing with multiple formats
    df['datetime'] = pd.to_datetime(
        df['datetime_str'],
        format="%m/%d/%y %I:%M %p",
        errors='coerce'    # If parsing fails → NaT (not error)
    )
    
    # ── Step 2: Extract date components ───────────────
    df['year']    = df['datetime'].dt.year
    df['month']   = df['datetime'].dt.month
    df['day']     = df['datetime'].dt.day
    df['hour']    = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.day_name()
    df['date_only'] = df['datetime'].dt.date

    # ── Step 3: Message text features ─────────────────
    # For media/deleted messages, text features = 0
    # Because "<Media omitted>" is not real user text

    df['message_length'] = df.apply(
        lambda row: len(str(row['message']))
        if row['message_type'] == 'normal'
        else 0,
        axis=1
    )

    df['word_count'] = df.apply(
        lambda row: count_words(row['message'])
        if row['message_type'] == 'normal'
        else 0,
        axis=1
    )

    df['emoji_count'] = df['message'].apply(count_emojis)
    # Note: we count emojis in ALL messages
    # because even media messages can have 
    # emoji captions sometimes

    # ── Step 4: URL features ───────────────────────────
    df['url_count'] = df['message'].apply(count_urls)
    df['has_url']   = df['url_count'] > 0

    # ── Step 5: Boolean flags ──────────────────────────
    df['is_media']   = df['message_type'] == 'media'
    df['is_deleted'] = df['message_type'] == 'deleted'

    # ── Step 6: Drop helper column ─────────────────────
    df = df.drop(columns=['datetime_str'])

    return df


# ── Quick Test ─────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.parser import parse_whatsapp_chat

    sample_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "sample", "sample_chat.txt"
    )

    print("Testing preprocessor...")
    df_raw  = parse_whatsapp_chat(sample_path)
    df_clean = preprocess(df_raw)

    print(f"✅ Shape after preprocessing : {df_clean.shape}")
    print(f"✅ Columns added             : "
          f"{list(df_clean.columns)}")
    print()
    print("Sample enriched data:")
    print(df_clean[['sender','message','hour',
                    'word_count','emoji_count']].head(5))