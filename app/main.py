# ============================================
# app/main.py
# Phase 7: Streamlit Application
#
# Purpose: Main entry point for the web app
# Run with: streamlit run app/main.py
# ============================================

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.parser       import parse_whatsapp_chat
from src.preprocessor import preprocess
from src.analyzer     import (
    get_basic_stats,
    get_user_stats,
    get_time_stats,
    get_content_stats
)
from src.nlp_engine   import (
    run_nlp_pipeline,
    get_sentiment_stats,
    get_tfidf_keywords
)
from src.visualizer   import (
    plot_user_activity,
    plot_hourly_activity,
    plot_daily_activity,
    plot_message_distribution,
    plot_wordcloud
)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title = "WhatsApp Intelligence Platform",
    page_icon  = "💬",
    layout     = "wide"
)

# ============================================
# SIDEBAR — File Upload
# ============================================
with st.sidebar:
    st.title("💬 WhatsApp Intelligence")
    st.markdown("---")
    st.subheader("📂 Upload Chat File")
    
    uploaded_file = st.file_uploader(
        label       = "Export your WhatsApp chat as .txt",
        type        = ["txt"],
        help        = "Go to WhatsApp → Chat → Export Chat → Without Media"
    )
    
    st.markdown("---")
    st.markdown("**How to export:**")
    st.markdown("1. Open WhatsApp chat")
    st.markdown("2. Tap ⋮ → More → Export Chat")
    st.markdown("3. Choose Without Media")
    st.markdown("4. Upload the .txt file here")

# ============================================
# MAIN CONTENT
# ============================================
if uploaded_file is None:
    # Welcome screen
    st.title("💬 WhatsApp Conversation Intelligence Platform")
    st.markdown("### Upload a WhatsApp chat export to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Chat Analytics**\nMessage stats, user activity, time patterns")
    with col2:
        st.info("🧠 **NLP Insights**\nSentiment analysis, keywords, word clouds")
    with col3:
        st.info("📈 **Visualizations**\nInteractive charts and heatmaps")

else:
    # ── Load and process data ──────────────────
    with st.spinner("🔄 Processing your chat..."):
        
        # Save uploaded file temporarily
        temp_path = "data/raw/uploaded_chat.txt"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run pipeline
        df_raw   = parse_whatsapp_chat(temp_path)
        df       = preprocess(df_raw)
        
        # Run analysis
        basic    = get_basic_stats(df)
        users    = get_user_stats(df)
        time     = get_time_stats(df)
        content  = get_content_stats(df)
        
        # Run NLP
        nlp_df    = run_nlp_pipeline(df)
        sentiment = get_sentiment_stats(nlp_df)
        keywords  = get_tfidf_keywords(nlp_df)

    st.success("✅ Chat processed successfully!")
    
    # ── Page Title ─────────────────────────────
    st.title("💬 WhatsApp Conversation Intelligence")
    st.markdown(f"**File:** {uploaded_file.name}")
    st.markdown("---")

    # ============================================
    # TAB LAYOUT
    # ============================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "📈 Visualizations", 
        "🧠 NLP Insights"
    ])

    # ── TAB 1: Overview ────────────────────────
    with tab1:
        st.header("📊 Chat Overview")
        
        # KPI Cards Row 1
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages",  basic['total_messages'])
        col2.metric("Participants",    basic['total_users'])
        col3.metric("Duration",        f"{basic['duration_days']} days")
        col4.metric("Total Words",     basic['total_words'])
        
        # KPI Cards Row 2
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total Emojis",    basic['total_emojis'])
        col6.metric("Media Shared",    basic['total_media'])
        col7.metric("Links Shared",    basic['total_links'])
        col8.metric("Peak Hour",       f"{time['peak_hour']}:00")
        
        st.markdown("---")
        
        # User Activity Table
        st.subheader("👥 User Activity Breakdown")
        st.dataframe(users, use_container_width=True)
        
        st.markdown("---")
        
        # Most active highlights
        col1, col2, col3 = st.columns(3)
        col1.success(f"🏆 Most Active: **{users.iloc[0]['sender']}**")
        col2.success(f"📅 Peak Day: **{time['peak_day']}**")
        col3.success(f"🕐 Peak Hour: **{time['peak_hour']}:00**")

    # ── TAB 2: Visualizations ──────────────────
    with tab2:
        st.header("📈 Visualizations")
        
        # Row 1: Bar + Pie
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_user_activity(users),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                plot_message_distribution(users),
                use_container_width=True
            )
        
        # Row 2: Hourly + Daily
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                plot_hourly_activity(time),
                use_container_width=True
            )
        with col4:
            st.plotly_chart(
                plot_daily_activity(time),
                use_container_width=True
            )
        
        # Row 3: Wordcloud (full width)
        st.subheader("☁️ Word Cloud")
        fig_wc = plot_wordcloud(content)
        st.pyplot(fig_wc)

    # ── TAB 3: NLP Insights ────────────────────
    with tab3:
        st.header("🧠 NLP Insights")
        
        # Sentiment Overview
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Sentiment Score", 
                    sentiment['overall_polarity'])
        
        dist = sentiment['sentiment_distribution']
        col2.metric("Positive Messages", 
                    int(dist.get('positive', 0)))
        col3.metric("Negative Messages", 
                    int(dist.get('negative', 0)))
        
        st.markdown("---")
        
        # Sentiment per user
        st.subheader("😊 Sentiment by User")
        st.dataframe(
            sentiment['user_sentiment'].reset_index(),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # TF-IDF Keywords
        st.subheader("🔑 Top Keywords per User (TF-IDF)")
        for user, kws in keywords.items():
            if kws:
                kw_str = " | ".join([f"**{w}** ({s})" for w, s in kws])
                st.markdown(f"**{user}:** {kw_str}")
        
        st.markdown("---")
        
        # Raw NLP data
        st.subheader("📝 Message-level Sentiment")
        st.dataframe(
            nlp_df[['sender','message',
                    'polarity','sentiment_label']],
            use_container_width=True
        )