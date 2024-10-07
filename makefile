
# Variables
BUCKET_NAME = # Bucket name

# Getting GCP credentials
get_credentials:
    gcloud auth application-default login

# Creating a new bucket
create_bucket:
    gsutil mb -l us-central1 gs://$(BUCKET_NAME)

# deploy the app with docker-compose
deploy:
    docker-compose up -d