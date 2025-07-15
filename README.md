# AI Interview System

This project is an AI-powered interview system designed to facilitate automated interviews using advanced language models and speech processing tools.

## Features
- Automated interview question generation
- Real-time audio processing
- Integration with ElevenLabs and Streamlit
- WebRTC support for live audio/video

## Requirements
Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App
To run the Streamlit-based interview app:

```bash
streamlit run beta_streamlit_app.py
```

Or, for the alternative app:

```bash
streamlit run src/beta_app.py
```

### Audio Processing
Some features require audio input/output. Ensure your microphone and speakers are configured properly.

## Project Structure
- `beta_streamlit_app.py`: Main Streamlit app
- `src/beta_app.py`: Alternative Streamlit app
- `corrected_interview.py`: Interview logic
- `notebooks/`: Jupyter notebooks for experiments
- `utils/`: Utility files and resources

## Notes
- Make sure to configure your `.env` file with the necessary API keys and environment variables.
- For ElevenLabs and OpenAI Whisper, you may need API access.

## License
This project is for educational and research purposes. 