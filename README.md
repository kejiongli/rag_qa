# step by step
Before start:
1. get your gcp project id (not the project name!)
2. install gcloud CLI locally and login `gcloud auth login`

On GCP
1. Enable GCP Artifact Registry API
1. Create a docker registry in a location (e.g. europe-west2)

LOCAL
1. authenticate docker: `gcloud auth configure-docker {location}-docker.pkg.dev`
2. build your docker image: `docker build Dockerfile --tag {TAG}`
3. tag the image: `docker tag {TAG} {location}-docker.pkg.dev/{GCP_PROJECT_ID}/{DOCKER REGISTRY NAME}/{TAG}`
4. push the image to gcp: `docker push {location}-docker.pkg.dev/{GCP_PROJECT_ID}/{DOCKER REGISTRY NAME}/{TAG}`

On GCP
1. Go to Artifact Registry, find the docker image you just pushed, click DEPLOY and select Cloud Run
2. In the Cloud Run config, make sure you set the following variables, then click CREATE
    * ingress control: All
    * container port: 8501
    * Environment Variable: 
      * GCLOUD_PROJECT_ID={your gcp project id}
      * BUCKET_NAME={the bucket where the pdf files are uploaded to}
    * \>=2 cpu cores, >=4G RAM
3. While waiting for the service starting, upload pdf files to cloud storage. 
4. Wait until your cloud run service starts, then click the auto-generated URL 