# ============================================
# src/analyzer.py
# Phase 2: Exploratory Data Analysis
#
# Purpose: Reusable analysis functions
# Called by: Streamlit app, notebooks
# ============================================

from collections import Counter
import pandas as pd
import numpy as np

def get_basic_stats(df):
    """Returns headline statistics about the chat."""
    user_df    = df[df['sender'] != 'System']
    first_date = df['datetime'].min()
    last_date  = df['datetime'].max()

    return {
        'total_messages' : len(df),
        'user_messages'  : len(user_df),
        'total_users'    : user_df['sender'].nunique(),
        'duration_days'  : (last_date - first_date).days,
        'first_date'     : first_date,
        'last_date'      : last_date,
        'total_words'    : int(df['word_count'].sum()),
        'total_emojis'   : int(df['emoji_count'].sum()),
        'total_media'    : int(df['is_media'].sum()),
        'total_links'    : int(df['has_url'].sum())
    }


def get_user_stats(df):
    """Returns per-user activity statistics."""
    user_df    = df[df['sender'] != 'System'].copy()

    user_stats = user_df.groupby('sender').agg(
        total_messages  = ('message',        'count'),
        total_words     = ('word_count',     'sum'),
        total_emojis    = ('emoji_count',    'sum'),
        total_media     = ('is_media',       'sum'),
        total_links     = ('has_url',        'sum'),
        avg_msg_length  = ('message_length', 'mean')
    ).reset_index()

    user_stats['message_percent'] = (
        user_stats['total_messages'] /
        user_stats['total_messages'].sum() * 100
    ).round(1)

    user_stats = user_stats.sort_values(
        'total_messages', ascending=False
    )
    return user_stats

def get_time_stats(df):
    """Returns time-based activity statistics."""
    
    day_order = ['Monday','Tuesday','Wednesday',
                 'Thursday','Friday','Saturday','Sunday']
    
    hourly  = df.groupby('hour')['message'].count()
    daily   = df.groupby('weekday')['message'].count()
    daily   = daily.reindex(day_order, fill_value=0)
    monthly = df.groupby('month')['message'].count()
    
    return {
        'hourly'         : hourly,
        'daily'          : daily,
        'monthly'        : monthly,
        'peak_hour'      : int(hourly.idxmax()),
        'peak_day'       : daily.idxmax(),
        'peak_month'     : int(monthly.idxmax())
    }


def get_content_stats(df):
    """Returns word frequency, emoji and media statistics."""
    
    normal_messages = df[df['message_type'] == 'normal']['message']
    all_text        = " ".join(normal_messages.astype(str))
    all_words       = all_text.lower().split()
    meaningful_words = [w for w in all_words if len(w) > 2]
    
    word_freq = Counter(meaningful_words)
    top_words = word_freq.most_common(15)
    
    user_df   = df[df['sender'] != 'System'].copy()
    
    emoji_stats = user_df.groupby('sender')['emoji_count'].sum().sort_values(ascending=False)
    media_stats = user_df.groupby('sender')['is_media'].sum().sort_values(ascending=False)
    
    return {
        'top_words'   : top_words,
        'word_freq'   : word_freq,
        'emoji_stats' : emoji_stats,
        'media_stats' : media_stats
    }
    
    

