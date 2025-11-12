# Web Application - True Random Number Generator

Flask-based web application for visualizing and processing data from a gamma-ray-based True Random Number Generator.

## Features

- **Real-time Data Visualization**: WebSocket-based live updates of gamma events
- **Interactive Q&A System**: Ask yes/no questions answered by quantum randomness
- **NIST Randomness Tests**: Statistical validation of random number quality
- **Rate Limiting**: Anti-spam protection
- **Docker Support**: Production-ready containerization

## Quick Start

### Using Docker (Recommended)

1. **Configure environment:**
   ```bash
   cp instance/.env.example instance/.env
   # Edit instance/.env with your settings
   ```

2. **Start the application:**
   ```bash
   docker-compose up -d
   ```

3. **Access:** http://localhost:5002

### Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure:**
   ```bash
   cp instance/.env.example instance/.env
   # Edit instance/.env
   ```

4. **Run:**
   ```bash
   python backend.py
   ```

5. **Access:** http://localhost:5000

## Configuration

Edit `instance/.env`:

```bash
# Secret token for API authentication
SECRET_TOKEN=your_secret_token_here

# CORS allowed origins (comma-separated)
CORS_ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:5000
```

**IMPORTANT:** Keep `instance/.env` secret! Never commit it to version control.

## API Endpoints

- `POST /upload` - Upload gamma event data (requires auth)
- `GET /ping` - Device health check (requires auth)
- `GET /nist_tests` - Run NIST randomness tests
- `GET /health` - Application health check
- `GET /latest_messages` - Get recent messages
- `GET /older_messages` - Paginated older messages

## Security

- Environment-based configuration (no hardcoded secrets)
- Token-based authentication
- Rate limiting (1 message per 30 seconds per IP)
- Profanity filtering
- Input sanitization
- CORS protection
- Non-root Docker user

## Docker Production Deployment

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build
```

## Database

The application uses SQLite with two main tables:
- `GammaEventEntry`: Stores gamma-ray detection events
- `ChatMessage`: Stores user questions and quantum answers

Database is automatically initialized on first run.

## Testing Randomness

Access the NIST test suite via:
- Web interface: Click "NIST Randomness Tests" button
- API: `GET /nist_tests`

Requires at least 1000 random bits for meaningful results.

## Development

The application runs on Flask with SocketIO for real-time updates. Key files:

- `backend.py` - Main Flask application
- `nist_randomness_tests.py` - Statistical test implementation
- `static/index.html` - Web interface
- `Dockerfile` - Production container
- `docker-compose.yml` - Orchestration

## Troubleshooting

**Database errors:**
- Ensure `instance/` directory exists and is writable
- Check disk space

**WebSocket issues:**
- Verify CORS settings in `.env`
- Check browser console for errors

**Authentication failures:**
- Ensure SECRET_TOKEN matches between firmware and backend
- Check request headers include Authorization token

## License

See main project README for license information.
