# For debugging: docker run -it --rm --entrypoint sh educ244-final-proj-container
FROM python:3.10.0-bullseye
WORKDIR /workspace

# Install dependencies first so Docker can reuse cached layers.
# RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code for the app into the image.
COPY ./app ./app

# Start the app when the container starts.
ENTRYPOINT [ "python", "./app/app.py" ]