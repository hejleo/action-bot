# Incari

This repository contains the Incari application setup using Docker Compose.

## Prerequisites

- Docker Engine installed (version 20.10.0 or higher)
- Docker Compose installed (version 2.0.0 or higher)
- Git (for cloning the repository)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd incari
```

### 2. Running the Application

First, build the Docker images:

```bash
docker compose build
```
This command builds or rebuilds all the services defined in your docker-compose.yml file. It creates Docker images based on the specifications in your Dockerfiles.

Then, start the application:

```bash
docker compose up
```
This command starts all services defined in your docker-compose.yml file. You'll see the logs from all containers in your terminal.

Alternatively, you can combine both commands:

```bash
docker compose up --build
```
This single command will both build the images and start the containers.

To run the application in the background (detached mode):

```bash
docker compose up -d
```

To stop the application:

```bash
docker compose down
```

### 3. Accessing the Application

Once the containers are running, open your web browser and navigate to:

```
http://localhost:3000
```

### 4. Viewing Logs

To view the logs while the application is running:

```bash
docker compose logs -f
```

## Troubleshooting

If you encounter issues:
1. Make sure all required ports are available
2. Verify Docker daemon is running
3. Try rebuilding the images with `docker compose build --no-cache`
4. Check the logs for error messages using `docker compose logs -f`