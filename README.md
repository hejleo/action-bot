# Incari

This repository contains the Incari application setup using Docker Compose.

## Prerequisites

- Docker Engine installed
- Docker Compose installed
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Running the Application

1. Build the services:
```bash
docker-compose build
```
This command will build the Docker images for all services.

2. Start the application:
```bash
docker-compose up
```
This command will start all services. You'll see the logs in your terminal.

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. To stop the application, press `Ctrl+C` in the terminal or run:
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
4. If you make changes to the code, run `docker-compose build` again



# Action Arranger Documentation

## Overview
The ActionArranger is a dual-model NLP system designed to interpret natural language commands and match them to appropriate node actions in a visual programming environment. It employs two distinct models for different aspects of text processing:

1. **FLAN-T5-XXL** - Primary language understanding model
2. **MiniLM-L12** - Lightweight semantic similarity model

## Architecture

### Language Processing Pipeline
1. User prompt ingestion
2. Action extraction and reordering (FLAN-T5)
3. Semantic similarity matching (MiniLM)
4. Node mapping and description generation

### Model Roles

#### FLAN-T5-XXL Model
- Primary text understanding and action extraction
- Handles temporal and logical ordering of actions
- **Current Limitations**: 
  - Model size may be excessive for the task
  - Processing speed could be improved
  - Consider replacing with a more specialized model

#### MiniLM-L12 Model
- Lightweight semantic similarity comparisons
- Used for matching actions to predefined nodes
- **Advantages**:
  - Fast processing
  - Low resource requirements
  - Sufficient for limited-scope matching

## Future Improvements

### Model Optimization
- Consider replacing FLAN-T5-XXL with a more focused model
- Potential alternatives:
  - BERT-based models for classification
  - Custom-trained models on node-specific data
  - Smaller T5 variants

### Architecture Enhancements
- Implement caching for frequent commands
- Add context awareness for better action interpretation
- Improve node matching accuracy with domain-specific training

## Resource Management
- Models are loaded with device-specific optimizations (MPS/CUDA/CPU)
- Implements memory-efficient inference
- Supports dynamic resource allocation

## Node System
- Organized in categories (Event, Action, Data nodes)
- Extensible through JSON definitions
- Supports rich metadata and descriptions

## Usage Considerations
- Best suited for prototype and development environments
- May require optimization for production deployment
- Consider hardware requirements for FLAN-T5-XXL model