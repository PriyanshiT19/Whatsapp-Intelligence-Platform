# ============================================
# src/visualizer.py
# Phase 3: Visualizations
#
# Purpose: All chart generation functions
# Called by: Streamlit app, notebooks
# Returns: plotly figures (interactive)
# ============================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_user_activity(user_stats):
    """Bar chart: Messages per user."""
    
    fig = px.bar(
        user_stats,
        x            = 'sender',
        y            = 'total_messages',
        color        = 'sender',
        text         = 'message_percent',
        title        = '📊 Messages per User',
        labels       = {'sender': 'User', 'total_messages': 'Messages'},
        color_discrete_sequence = px.colors.qualitative.Set2
    )
    
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=False, plot_bgcolor='white')
    
    return fig


def plot_hourly_activity(time_stats):
    """Bar chart: Messages by hour of day."""
    
    hourly = time_stats['hourly'].reset_index()
    hourly.columns = ['hour', 'messages']
    
    fig = px.bar(
        hourly,
        x     = 'hour',
        y     = 'messages',
        title = '🕐 Activity by Hour of Day',
        labels = {'hour': 'Hour (24h)', 'messages': 'Messages'},
        color = 'messages',
        color_continuous_scale = 'Blues'
    )
    
    fig.update_layout(plot_bgcolor='white')
    
    return fig

def plot_daily_activity(time_stats):
    """Bar chart: Messages by day of week."""
    
    daily = time_stats['daily'].reset_index()
    daily.columns = ['day', 'messages']
    
    fig = px.bar(
        daily,
        x     = 'day',
        y     = 'messages',
        title = '📅 Activity by Day of Week',
        labels = {'day': 'Day', 'messages': 'Messages'},
        color = 'messages',
        color_continuous_scale = 'Greens'
    )
    
    fig.update_layout(plot_bgcolor='white')
    
    return fig


def plot_message_distribution(user_stats):
    """Pie chart: Message share per user."""
    
    fig = px.pie(
        user_stats,
        values = 'total_messages',
        names  = 'sender',
        title  = '🥧 Message Share by User',
        color_discrete_sequence = px.colors.qualitative.Set2
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def plot_wordcloud(content_stats):
    """Wordcloud from top words."""
    
    word_freq = dict(content_stats['word_freq'])
    
    wc = WordCloud(
        width            = 800,
        height           = 400,
        background_color = 'white',
        colormap         = 'viridis',
        max_words        = 100
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('☁️ Most Used Words', fontsize=16, pad=20)
    
    return fig

