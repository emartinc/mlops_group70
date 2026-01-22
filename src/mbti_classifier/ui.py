import os

import plotly.graph_objects as go
import requests
import streamlit as st

# API Configuration - use environment variable if available (for Docker), otherwise localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# MBTI Type Descriptions
MBTI_DESCRIPTIONS = {
    "INTJ": "The Architect - Strategic, logical, and innovative thinkers",
    "INTP": "The Logician - Innovative inventors with an unquenchable thirst for knowledge",
    "ENTJ": "The Commander - Bold, imaginative and strong-willed leaders",
    "ENTP": "The Debater - Smart and curious thinkers who cannot resist an intellectual challenge",
    "INFJ": "The Advocate - Quiet and mystical, yet very inspiring and tireless idealists",
    "INFP": "The Mediator - Poetic, kind and altruistic people, always eager to help a good cause",
    "ENFJ": "The Protagonist - Charismatic and inspiring leaders, able to mesmerize their listeners",
    "ENFP": "The Campaigner - Enthusiastic, creative and sociable free spirits",
    "ISTJ": "The Logistician - Practical and fact-minded individuals, whose reliability cannot be doubted",
    "ISFJ": "The Defender - Very dedicated and warm protectors, always ready to defend their loved ones",
    "ESTJ": "The Executive - Excellent administrators, unsurpassed at managing things or people",
    "ESFJ": "The Consul - Extraordinarily caring, social and popular people, always eager to help",
    "ISTP": "The Virtuoso - Bold and practical experimenters, masters of all kinds of tools",
    "ISFP": "The Adventurer - Flexible and charming artists, always ready to explore and experience something new",
    "ESTP": "The Entrepreneur - Smart, energetic and very perceptive people, who truly enjoy living on the edge",
    "ESFP": "The Entertainer - Spontaneous, energetic and enthusiastic people – life is never boring around them",
}


def create_radar_plot(probabilities: dict) -> go.Figure:
    """
    Create a radar plot showing personality dimension scores.

    Args:
        probabilities: Dictionary with probability scores for each dimension

    Returns:
        Plotly figure object
    """
    # Extract scores for visualization
    # We'll show the probability of each trait (E, S, T, J vs I, N, F, P)
    categories = ["Extraversion", "Sensing", "Thinking", "Judging"]

    # Get the scores (using the higher probability for each dimension)
    scores = [
        max(probabilities["E"], probabilities["I"]) * 100,
        max(probabilities["S"], probabilities["N"]) * 100,
        max(probabilities["T"], probabilities["F"]) * 100,
        max(probabilities["J"], probabilities["P"]) * 100,
    ]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=categories,
            fill="toself",
            name="Personality Profile",
            line=dict(color="#1f77b4", width=2),
            fillcolor="rgba(31, 119, 180, 0.3)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", showticklabels=True),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        title=dict(text="MBTI Personality Dimensions", x=0.5, xanchor="center", font=dict(size=20)),
        height=500,
        margin=dict(t=80, b=40, l=80, r=80),
    )

    # Add custom annotations for better readability
    annotations_text = "<br>".join(
        [
            f"<b>{cat}</b>: {lbl}"
            for cat, lbl in zip(
                ["E/I", "S/N", "T/F", "J/P"],
                [
                    f"{'Extraversion' if probabilities['E'] > probabilities['I'] else 'Introversion'}",
                    f"{'Sensing' if probabilities['S'] > probabilities['N'] else 'Intuition'}",
                    f"{'Thinking' if probabilities['T'] > probabilities['F'] else 'Feeling'}",
                    f"{'Judging' if probabilities['J'] > probabilities['P'] else 'Perceiving'}",
                ],
            )
        ]
    )

    return fig


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="MBTI Personality Classifier", page_icon="🧠", layout="wide")

    # Header
    st.title("🧠 MBTI Personality Classifier")
    st.markdown(
        """
    This app predicts your Myers-Briggs personality type based on your writing style.
    Enter some text (the more the better!) and discover your personality dimensions.
    """
    )

    # Check API health
    if not check_api_health():
        st.error(
            """
        ⚠️ **API Server is not running!**
        
        Please start the API server first:
        ```bash
        uv run uvicorn mbti_classifier.api:app --reload --host 0.0.0.0 --port 8000
        ```
        """
        )
        st.stop()

    # Sidebar with information
    with st.sidebar:
        st.header("About MBTI")
        st.markdown(
            """
        **Myers-Briggs Type Indicator (MBTI)** is a personality assessment 
        that categorizes people into 16 personality types based on four dimensions:
        
        - **E/I**: Extraversion vs Introversion
        - **S/N**: Sensing vs Intuition
        - **T/F**: Thinking vs Feeling
        - **J/P**: Judging vs Perceiving
        """
        )

        st.header("Tips for Best Results")
        st.markdown(
            """
        - Provide **at least 100-200 words** of your writing
        - Use **natural, conversational text**
        - Include personal thoughts and opinions
        - Mix different topics if possible
        """
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Your Text")

        # Text input
        text_input = st.text_area(
            label="Write something about yourself, your thoughts, or any topic...",
            height=300,
            placeholder="Example: I love spending time alone with my thoughts. I enjoy analyzing complex problems "
            "and coming up with creative solutions. I prefer to plan things ahead rather than being spontaneous...",
        )

        # Character count
        char_count = len(text_input)
        word_count = len(text_input.split())

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Characters: {char_count}")
        with col_b:
            st.caption(f"Words: {word_count}")

        # Predict button
        predict_button = st.button("🔮 Predict Personality Type", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        results_placeholder = st.empty()

        if not predict_button:
            results_placeholder.info("👈 Enter your text and click 'Predict' to see results")

    # Make prediction when button is clicked
    if predict_button:
        if len(text_input.strip()) < 50:
            st.error("⚠️ Please provide at least 50 characters of text for accurate prediction.")
        else:
            with st.spinner("🧠 Analyzing your personality..."):
                try:
                    # Call API
                    response = requests.post(f"{API_URL}/predict", json={"text": text_input}, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        mbti_type = result["mbti_type"]
                        probabilities = result["probabilities"]

                        # Display results in the right column
                        with col2:
                            # MBTI Type
                            st.success(f"### Your Personality Type: **{mbti_type}**")

                            # Description
                            if mbti_type in MBTI_DESCRIPTIONS:
                                st.info(f"*{MBTI_DESCRIPTIONS[mbti_type]}*")

                            # Dimension breakdown
                            st.markdown("#### Dimension Breakdown:")
                            for dim in result["dimensions"]:
                                percentage = dim["score"] * 100
                                label = dim["label"]
                                dimension = dim["dimension"]

                                # Determine opposite label
                                opposites = {
                                    "E": "I",
                                    "I": "E",
                                    "S": "N",
                                    "N": "S",
                                    "T": "F",
                                    "F": "T",
                                    "J": "P",
                                    "P": "J",
                                }
                                opposite = opposites.get(label, "")

                                # Color code based on strength
                                if percentage > 70:
                                    color = "🟢"
                                elif percentage > 55:
                                    color = "🟡"
                                else:
                                    color = "🟠"

                                st.markdown(
                                    f"{color} **{dimension}**: {label} ({percentage:.1f}%) vs {opposite} ({100-percentage:.1f}%)"
                                )

                        # Radar plot (full width below)
                        st.markdown("---")
                        st.subheader("📊 Personality Profile Visualization")
                        fig = create_radar_plot(probabilities)
                        st.plotly_chart(fig, use_container_width=True)

                        # Additional insights
                        st.markdown("---")
                        st.subheader("💡 Understanding Your Results")

                        insights_col1, insights_col2 = st.columns(2)

                        with insights_col1:
                            st.markdown("**Energy Source**")
                            if probabilities["E"] > probabilities["I"]:
                                st.write("🌟 You gain energy from external interactions (Extraversion)")
                            else:
                                st.write("🧘 You gain energy from internal reflection (Introversion)")

                            st.markdown("**Information Processing**")
                            if probabilities["S"] > probabilities["N"]:
                                st.write("📋 You focus on concrete facts and details (Sensing)")
                            else:
                                st.write("🔮 You focus on patterns and possibilities (Intuition)")

                        with insights_col2:
                            st.markdown("**Decision Making**")
                            if probabilities["T"] > probabilities["F"]:
                                st.write("🧠 You make decisions based on logic (Thinking)")
                            else:
                                st.write("❤️ You make decisions based on values (Feeling)")

                            st.markdown("**Lifestyle Approach**")
                            if probabilities["J"] > probabilities["P"]:
                                st.write("📅 You prefer structure and planning (Judging)")
                            else:
                                st.write("🎨 You prefer flexibility and spontaneity (Perceiving)")

                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The text might be too long. Try with shorter text.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to API. Make sure the API server is running.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption(
        """
    **Note**: This is an AI-based prediction and should be used for entertainment and self-reflection purposes. 
    For a comprehensive personality assessment, consider taking an official MBTI test.
    """
    )


if __name__ == "__main__":
    main()