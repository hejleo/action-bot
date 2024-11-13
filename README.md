# Incari

This repository contains the Incari application setup using Docker Compose.

## Prerequisites

- Docker Engine installed
- Docker Compose installed
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Running the Application

1. Start the application:
```bash
docker-compose up
```
This command will build and start all services. You'll see the logs in your terminal.

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. To stop the application, press `Ctrl+C` in the terminal or run:
```bash
docker-compose down
```

## Viewing Logs

While the application is running, view the logs with:
```bash
docker-compose logs -f
```

## Troubleshooting

If you encounter issues:
1. Make sure port 5000 is available
2. Verify Docker daemon is running
3. Check the logs for error messages