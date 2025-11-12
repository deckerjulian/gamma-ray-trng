# Gamma Ray True Random Number Generator (GR-TRNG)

A hardware-based True Random Number Generator using gamma-ray detection with a dual-core RP2040 microcontroller and a Flask-based web application for visualization and randomness testing.

## 🌟 Overview

This project combines quantum physics and modern web technology to create a true random number generator. It uses gamma-ray detection events from radioactive decay to generate cryptographically secure random numbers. The system consists of two main components:

1. **Firmware** - MicroPython code running on RP2040 (W5500-EVB-Pico) that collects gamma-ray events
2. **Web Application** - Flask-based server that processes and visualizes the random data

## 🎯 Features

### Firmware (RP2040 Dual-Core)
- **Dual-Core Architecture**: Core 0 handles networking/uploads, Core 1 dedicated to high-speed ADC data collection
- **Real-time Gamma Event Detection**: Monitors ADC for gamma-ray events above configurable threshold
- **Automated Data Upload**: Buffers and uploads event data to backend server
- **Web Interface**: Built-in status page showing device health and performance metrics
- **Watchdog Protection**: Hardware watchdog ensures system stability
- **DHCP/Static IP Support**: Flexible network configuration

### Web Application
- **Real-time Data Visualization**: WebSocket-based live updates of gamma events
- **Interactive Q&A System**: Ask yes/no questions answered by quantum randomness
- **NIST Randomness Tests**: Built-in statistical validation of random number quality
- **Data Persistence**: SQLite database for storing events and messages
- **Profanity Filtering**: Content moderation for user input
- **Rate Limiting**: Anti-spam protection
- **Docker Support**: Easy deployment with Docker Compose

## 📁 Project Structure

```
gamma-ray-trng/
├── firmware/                    # MicroPython firmware for RP2040
│   ├── main.py                 # Main firmware code
│   ├── uping.py                # Network ping utility
│   ├── config.example.py       # Configuration template
│   └── README.md               # Firmware documentation
│
├── webapplication/             # Flask web application
│   ├── backend.py              # Main Flask application
│   ├── nist_randomness_tests.py # NIST statistical test suite
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker container definition
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── instance/               # Instance-specific files (not in git)
│   │   ├── .env                # Backend config (gitignored)
│   │   ├── data.db             # SQLite database (gitignored)
│   │   └── random_bits.txt     # Generated random bits
│   └── static/                 # Web interface assets
│       ├── index.html          # Main web interface
│       ├── css/                # Stylesheets
│       ├── js/                 # JavaScript libraries
│       └── fonts/              # Web fonts
```

## 🚀 Quick Start

### Prerequisites

**For Firmware:**
- Raspberry Pi Pico with W5500 Ethernet module (W5500-EVB-Pico)
- MicroPython installed on the device
- ADC connected to gamma-ray detector circuit (Pin 26)

**For Web Application:**
- Python 3.11+
- Docker and Docker Compose (optional, recommended)
- Or: pip and virtualenv for local development

### Setup - Firmware

1. **Configure the device:**
   ```bash
   cd firmware
   cp config.example.py config.py
   ```

2. **Edit `config.py` with your settings:**
   ```python
   HOST = "your-server.example.com"
   AUTH_TOKEN = "your-secret-token-here"
   NETWORK_IP = "192.168.1.100"
   NETWORK_GATEWAY = "192.168.1.1"
   NETWORK_DNS = "8.8.8.8"
   ```

3. **Upload to your device:**
   - Upload `main.py`, `config.py`, and `uping.py` to your RP2040
   - Reboot the device

### Setup - Web Application

#### Option 1: Docker (Recommended)

1. **Configure environment:**
   ```bash
   cd webapplication
   cp instance/.env.example instance/.env
   ```

2. **Edit `instance/.env`:**
   ```bash
   SECRET_TOKEN=your-secret-token-here
   CORS_ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:5000
   ```

3. **Start the application:**
   ```bash
   docker-compose up -d
   ```

4. **Access the web interface:**
   - Open `http://localhost:5002` in your browser

#### Option 2: Local Development

1. **Create virtual environment:**
   ```bash
   cd webapplication
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp instance/.env.example instance/.env
   # Edit instance/.env with your settings
   ```

4. **Run the application:**
   ```bash
   python backend.py
   ```

5. **Access the web interface:**
   - Open `http://localhost:5000` in your browser

## 🔐 Security

### Important Security Notes

⚠️ **NEVER commit sensitive data to version control!**

The following files contain sensitive information and are gitignored:
- `firmware/config.py` - Device credentials and network settings
- `webapplication/instance/.env` - Server secrets and API tokens
- `webapplication/instance/*.db` - Database files

### Generating Secure Tokens

Generate a strong authentication token:

```bash
# Linux/macOS
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or use openssl
openssl rand -hex 32
```

### Security Best Practices

1. **Use HTTPS**: Always use HTTPS in production
2. **Strong Tokens**: Use cryptographically secure random tokens (≥32 bytes)
3. **Firewall**: Restrict access to backend server
4. **Rate Limiting**: Already implemented, but monitor for abuse
5. **Docker User**: Application runs as non-root user (UID 1000)

## 📊 NIST Randomness Tests

The application includes a comprehensive NIST SP 800-22 statistical test suite for validating randomness quality:

- **Frequency (Monobit) Test**
- **Block Frequency Test**
- **Runs Test**
- **Longest Run of Ones Test**
- **Binary Matrix Rank Test**
- **Discrete Fourier Transform Test**
- **Non-overlapping Template Matching Test**
- **Overlapping Template Matching Test**
- **Maurer's Universal Statistical Test**
- **Linear Complexity Test**
- **Serial Test**
- **Approximate Entropy Test**
- **Cumulative Sums Test**

Access tests via the web interface or API endpoint: `/nist_tests`

## 🔧 Configuration

### Firmware Configuration

Edit `firmware/config.py`:

```python
# Server Configuration
HOST = "your-server.example.com"
AUTH_TOKEN = "your-secret-token"

# Network Configuration (DHCP is attempted first, these are fallback)
NETWORK_IP = "192.168.1.100"
NETWORK_MASK = "255.255.255.0"
NETWORK_GATEWAY = "192.168.1.1"
NETWORK_DNS = "8.8.8.8"
```

### Web Application Configuration

Edit `webapplication/instance/.env`:

```bash
# Secret token for API authentication (must match firmware)
SECRET_TOKEN=your_secret_token_here

# CORS allowed origins (comma-separated)
CORS_ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:5000
```

## 📡 API Endpoints

### Data Upload (POST /upload)
Upload gamma-ray event data from device.

**Headers:**
- `Authorization: <AUTH_TOKEN>`
- `Content-Type: application/json`

**Body:**
```json
{
  "timestamp": 1234567890,
  "data": [100, 150, 200, ...],
  "device_id": "trng_001",
  "buffer_size": 500
}
```

### Ping (GET /ping)
Health check endpoint for device status.

**Headers:**
- `Authorization: <AUTH_TOKEN>`
- `Content-Type: application/json`

**Body (optional):**
```json
{
  "timestamp": "2025-11-12 10:30:00"
}
```

### NIST Tests (GET /nist_tests)
Run NIST randomness tests on collected data.

### Health Check (GET /health)
Simple health check for monitoring.

## 🐳 Docker Deployment

The application includes production-ready Docker configuration:

- **Multi-stage build**: Optimized image size
- **Non-root user**: Security best practice
- **Health checks**: Automatic container monitoring
- **Volume mounting**: Persistent data storage
- **Log rotation**: Automatic log management

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild after changes
docker compose up -d --build
```

## 🔍 Monitoring

### Device Web Interface

The RP2040 firmware includes a built-in web interface accessible at:
`http://<device-ip>/`

Features:
- Real-time sampling rate
- Upload statistics
- Queue status
- Memory usage
- Error log
- Device restart

### Application Monitoring

Monitor the Flask application:
- Health endpoint: `/health`
- Last device ping: Displayed on main page
- Database query performance
- WebSocket connection status

## 🛠️ Troubleshooting

### Firmware Issues

**Device not connecting to network:**
- Check `config.py` settings
- Verify network cable connection
- Check DHCP server logs
- Try static IP configuration

**Data not uploading:**
- Verify `AUTH_TOKEN` matches between firmware and backend
- Check firewall rules
- Monitor device web interface for errors
- Check backend server logs

### Web Application Issues

**Database errors:**
- Ensure `instance/` directory has write permissions
- Check disk space
- Verify SQLite is installed

**WebSocket not connecting:**
- Check CORS settings in `.env`
- Verify firewall allows WebSocket connections
- Check browser console for errors

## 📈 Performance

### Firmware Performance (RP2040)
- **Sampling Rate**: ~100 kHz (dependent on ADC configuration)
- **Core 0**: Network, uploads, web server (non-critical path)
- **Core 1**: Dedicated high-speed ADC sampling (critical path)
- **Upload Queue**: Configurable (default: 50 buffers)
- **Memory**: ~180 KB available after initialization

### Web Application Performance
- **Database**: SQLite with automatic cleanup
- **WebSocket**: Real-time updates to all connected clients
- **Rate Limiting**: 1 message per 30 seconds per IP
- **Queue Management**: Automatic old data cleanup

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **NIST**: For the SP 800-22 statistical test suite
- **MicroPython**: For embedded Python support
- **Flask**: For the web framework
- **Bootstrap**: For UI components
- **Socket.IO**: For real-time communication

## ⚠️ Disclaimer

This random number generator is designed for experimental and educational purposes. While it uses true quantum randomness from radioactive decay, proper validation and testing should be performed before using it in cryptographic or security-critical applications.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with quantum randomness and modern web technology** 🎲⚛️
