# Use the official Rasa image as the base image
FROM rasa/rasa:latest-full

# Set the working directory to the Rasa project directory
WORKDIR /app

# Copy the Rasa project files into the container
COPY . /app

USER root
RUN chown -R rasa:rasa /app
USER rasa
# train model
#RUN [ -f ".rasa" ] rm -r .rasa/
#RUN [ -f "models" ] rm -r models
RUN rasa train

# Expose the port on which Rasa will run
EXPOSE 5005

# Start both Rasa server and actions server
ENTRYPOINT ["rasa"]
CMD ["run", "-m", "models", "--enable-api", "--cors", "*"]


