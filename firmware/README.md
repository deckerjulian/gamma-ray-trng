# TRNG Firmware

MicroPython firmware for the True Random Number Generator device.

## Setup

1. Copy the configuration template:
   ```bash
   cp config.example.py config.py
   ```

2. Edit `config.py` with your actual values:
   - `HOST`: Your backend server hostname
   - `AUTH_TOKEN`: Secret authentication token (must match backend)
   - `NETWORK_IP`: Static IP address for the device
   - `NETWORK_GATEWAY`: Your network gateway
   - `NETWORK_DNS`: DNS server

3. Upload to your device:
   ```bash
   # Upload main.py and config.py to your MicroPython device
   ```

## Security

**IMPORTANT:** Never commit `config.py` to version control! This file contains sensitive credentials.

## Configuration

- `config.example.py`: Template with placeholder values (safe to commit)
- `config.py`: Your actual configuration (ignored by git)
