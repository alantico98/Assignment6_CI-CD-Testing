# Movie Review Sentiment Analyzer

This project implements a multi-container MLOps system that:

* Serves sentiment predictions, either **positive** or **negative**, via a FastAPI service
* Monitors model behavior and data drift in real-time using a Streamlit dashboard
* Shares prediction logs using a Docker volume, accessible to both containers
* Includes a script that uses a test file to systematically evaluate model performance via the API
* Runs on a publicly accessible AWS EC2 instance

## Prerequisites

Before running this app, make sure you have the following installed:

### 1. Python 3.13+

You can check your version with:

```bash
python --version
```

### 2. pip (Python package manager)

```bash
pip --version
```

### 3. Git

```bash
git --version
```

### 4. WSL 2 Backend (Windows users only, and if ssh'ing through a terminal)

```bash
wsl --version
```

### 5. Docker Installed (required)

```bash
docker --version
```

## For Local Development

Follow these steps to run the app locally (using Docker):

1. Clone the repo using bash (if on Windows, use WSL that allows for ssh cloning):

```bash
bash

git clone git@github.com:alantico98/Assignment5_ModelMonitoring.git

cd Assignment5_ModelMonitoring
```

2. Build the image (using WSL or Mac)

```bash
make build
```

4. Run the Container (using WSL or Mac)

```bash
make run
```

This will:
* Start the FastAPI app on http://localhost:8000
* Start the Streamlit Dashboard on http://localhost:8501.
* Create a named Docker volume sentiment-logs

If using Postman Desktop:
* Send a request to http://0.0.0.0:8000/health to check that the server is running and healthy

To use **curl** to interact with the API:
* Health Check:

```bash
curl -X GET http://localhost:8000/health
```

* Predict Sentiment"

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I loved this movie!", "true_label": "positive"}'
```

* Example response: {"sentiment": "positive"}

5. (Optional) Run the evaluation script

From ./Assignment5_ModelMonitoring:

```bash
python evaluate.py
```

This will request responses from the API server via JSON body objects, using the test.json file

6. Clean Up (using WSL or Mac)

```bash
make clean
```

## How to Deploy on an EC2 Instance

Follow these steps to run the app on an EC2 instance:

1. Navigate to AWS Console and Open an EC2 Page
2. Select "Launch Instance"
	- Choose a name for your web server. Ex: "My Streamlit Application"
	- Under "Application and OS Images", select "Ubuntu". The default Ubuntu 24.04 LTS version is fine
	- Choose the architecture that suits your machine type (if not defaulted to the correct one already)
	- Under "Instance Type", select "t2.micro"
	- Create a key pair so that you can ssh into the EC2 instance:
		* Create a name
		* Select "RSA" and have the private key file format in ".pem"
		* Store in a secure location for use later
	- Under the "Security" tab, select the hyperlink for your instance under "Security groups"
	- Once inside your security group, under "Inbound rules", select "Edit Inbound Rules":
		1. Add rule for FastAPI backend
			* Port range=8000, Source "Anywhere (0.0.0.0/0)"
		2. Add rule for Streamlit FrontEnd
			* Port range=8501, Source "Anywhere (0.0.0.0/0)"1. Clone the repo using bash (if on Windows, use WSL that allows for ssh cloning):

3. Connect to your EC2 instance
	- If using AWS, select your instance under "Instances" and then select "Connect" via Public IP
	- If SSH'ing (using the Key Pair you generated)
	    1. ssh -i /path/to/your/key.pem ubuntu@<EC2_PUBLIC_IPv4_ADDRESS>
	    2. If you get a "WARNING: UNPROTECTED PRIVATE KEY FILE", the following should fix it
		    1. Try moving your key to your .ssh directory (ex: "mv /path/to/your/key.pem ~/.ssh")
		    2. Restrict directory $ file permissions:
            ```bash
			chmod 700 ~/.ssh
			chmod 600 ~/.ssh/key.pem
            ```
		    3. Ensure you own the file
			```bash
            chown $USER:$USER ~/.ssh/streamlit_app_key.pem"
            ```
		4. Try connecting again it should be successful
			```bash
            ssh -i ~/.ssh/key.pem ubuntu@<EC2_PUBLIC_IPv4_ADDRESS>
            ```

4. Install Docker on the server (once)
```bash
# On EC2
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Let your user run docker without sudo
sudo usermod -aG docker $USER
# re-login to pick up group membership
exit
# log back in:
ssh -i /path/to/key.pem ubuntu@EC2_PUBLIC_DNS

# Verify installation
docker version
docker compose version
git version

# Check the Docker daemon is running
sudo systemctl status docker --no-pager

# (If Docker is installed but not enabled, start and enable it)
sudo systemctl start docker
sudo systemctl enable docker
```

5. Clone the repo to your EC2 instance
```bash
git clone https://github.com/alantico98/Assignment6_CI-CD-Testing.git
cd Assignment6_CI-CD-Testing

# Without merging, git checkout the dev branch
git checkout dev
```

6. Deploy the containers in detached mode to build and run the containers
```bash
# Build the containers
docker build -t sentiment-api ./api
docker build -t sentiment-monitor ./monitoring

# Create a Docker Network and Volume for the containers
docker network create sentiment-net || true
docker volume create sentiment-logs || true

# Run
docker run -d --rm --name api \
--network sentiment-net \
-v sentiment-logs:/logs \
-p 8000:8000 \
sentiment-api

# Run the streamlit monitoring dashboard
docker run -d --rm --name monitor \
--network sentiment-net \
-v sentiment-logs:/logs \
-p 8501:8501 \
sentiment-monitor
```

7. Access the applications using your EC2 instance Public IP

    * FastAPI docs: https://<EC2_PUBLIC_IP>:8000/docs
    * Streamlit dashboard: https://<EC2_PUBLIC_IP>:8501

8. When finished, run the following to close and clean up the docker images
```bash
# Clean up the containers
docker stop api monitor || true
docker network rm $(NETWORK_NAME) || true
docker volume rm $(VOLUME_NAME) || true
docker rmi $(APP_NAME_API) $(APP_NAME_MONITORING) || true
```
