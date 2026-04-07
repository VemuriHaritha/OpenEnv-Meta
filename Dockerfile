# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11

# Set up a new user named "user" with user ID 1000 (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Create __init__ files so Python treats dirs as packages
RUN touch data/__init__.py tasks/__init__.py

# Expose port for HF Spaces (always 7860)
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "app.py"]
