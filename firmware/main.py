import network
import ntptime
import machine
from machine import Pin, SPI, WDT
import utime
import _thread
import urequests
import uping
import gc
import socket
import json
import sys

# Import sensitive configuration from external file
try:
    from config import HOST, AUTH_TOKEN, NETWORK_IP, NETWORK_MASK, NETWORK_GATEWAY, NETWORK_DNS
except ImportError:
    print("ERROR: config.py not found!")
    print("Please copy config.example.py to config.py and configure it.")
    raise

# Configuration Constants
class Config:
    # Sensitive data from config.py
    HOST = HOST
    AUTH_TOKEN = AUTH_TOKEN
    NETWORK_IP = NETWORK_IP
    NETWORK_MASK = NETWORK_MASK
    NETWORK_GATEWAY = NETWORK_GATEWAY
    NETWORK_DNS = NETWORK_DNS
    
    # Public endpoints
    ENDPOINT = "upload"
    PING_ENDPOINT = "ping"
    
    # ADC Configuration
    ADC_PIN = 26
    THRESHOLD_VOLTAGE_MV = 120
    BUFFER_SIZE = 500
    ADC_REFERENCE_VOLTAGE = 3.3
    ADC_RESOLUTION = 65535
    
    # Timing Configuration
    WATCHDOG_TIMEOUT_MS = 8000  # RP2040 maximum is ~8388ms
    UPLOAD_THROTTLE_MS = 50     # Even faster: 50ms for aggressive uploads
    PING_INTERVAL_SECONDS = 300 # Send ping every 5 minutes (300 seconds)
    MAX_CONNECTION_ATTEMPTS = 10
    MAX_NTP_ATTEMPTS = 5
    NTP_RETRY_DELAY_BASE = 2
    
    # Queue Configuration
    MAX_QUEUE_SIZE = 50         # Increased from 10 to 50 to handle bursts
    
    # Web Interface Configuration
    WEB_PORT = 80
    WEB_HOST = '0.0.0.0'
    WEB_ENABLED = True

class TRNGDevice:
    def __init__(self):
        self.adc = machine.ADC(Config.ADC_PIN)
        self.samples = [0] * Config.BUFFER_SIZE
        self.data_queue = []
        self.running = True
        self.wdt = WDT(timeout=Config.WATCHDOG_TIMEOUT_MS)
        self.threshold_value = self._calculate_threshold()
        self.restart_requested = False
        
        # Dual-core synchronization variables
        self.core1_running = False
        self.trigger_found = False
        self.trigger_time = 0
        self.current_index = 0
        self.buffer_ready = False
        self.core1_error = None
        
        # Status tracking for web interface
        self.status = {
            'uptime_start': utime.time(),
            'samples_collected': 0,
            'triggers_found': 0,
            'uploads_successful': 0,
            'uploads_failed': 0,
            'queue_overflows': 0,
            'last_trigger_time': 0,
            'last_upload_time': 0,
            'last_ping_time': 0,
            'ping_successful': 0,
            'ping_failed': 0,
            'network_connected': False,
            'ntp_synchronized': False,
            'current_queue_size': 0,
            'memory_free': 0,
            'errors': []
        }
        
    def _calculate_threshold(self):
        """Convert threshold voltage from mV to ADC units"""
        voltage_ratio = (Config.THRESHOLD_VOLTAGE_MV / 1000.0) / Config.ADC_REFERENCE_VOLTAGE
        return int(voltage_ratio * Config.ADC_RESOLUTION)
    
    def check_threshold(self, sample):
        """Optimized threshold check using integer comparison"""
        return sample > self.threshold_value
    
    def add_error(self, error_msg):
        """Add error to status tracking with timestamp"""
        error_entry = {
            'timestamp': utime.time(),
            'message': str(error_msg)
        }
        self.status['errors'].append(error_entry)
        # Keep only last 10 errors
        if len(self.status['errors']) > 10:
            self.status['errors'].pop(0)
    
    def update_memory_status(self):
        """Update memory usage information"""
        try:
            gc.collect()
            # MicroPython specific memory info
            import micropython
            self.status['memory_free'] = micropython.mem_info()[0] if hasattr(micropython, 'mem_info') else 0
        except:
            self.status['memory_free'] = 0
    
    def stop(self):
        """Stop all operations and clean up resources"""
        print("Stopping device...")
        self.running = False
        self.core1_running = False
        
        # Give operations time to finish
        utime.sleep(2)
        
        # Force garbage collection to free memory
        gc.collect()
        print("Device stopped")
    
    def _core1_data_collection(self):
        """Core 1: Dedicated high-frequency data collection
        
        This function runs on the second CPU core and is responsible
        only for collecting ADC samples and detecting triggers.
        It runs at maximum speed without any blocking operations.
        """
        print("Core 1: Starting dedicated data collection")
        self.core1_running = True
        
        # Local variables for maximum performance
        local_samples = [0] * Config.BUFFER_SIZE
        local_index = 0
        local_trigger_found = False
        local_trigger_time = 0
        sample_count = 0
        loop_counter = 0  # Debug counter
        
        try:
            while self.core1_running and self.running:
                loop_counter += 1
                
                # Debug output every 100k samples
                if loop_counter % 100000 == 0:
                    print(f"Core 1: Processed {loop_counter} samples, running: {self.core1_running}")
                    # Update samples collected more frequently for web interface
                    self.status['samples_collected'] += 100000
                
                # High-speed ADC reading - this is the only operation on Core 1
                try:
                    sample = self.adc.read_u16()
                except Exception as adc_error:
                    print(f"Core 1: ADC read error: {adc_error}")
                    self.core1_error = f"ADC read error: {adc_error}"
                    break
                    
                local_samples[local_index] = sample
                sample_count += 1
                
                # Threshold detection
                if not local_trigger_found and self.check_threshold(sample):
                    local_trigger_time = utime.ticks_ms()  # Use ticks for higher precision
                    local_trigger_found = True
                    print(f"Core 1: Trigger detected at index {local_index}")
                
                # Buffer completion check
                if local_trigger_found and local_index == Config.BUFFER_SIZE - 1:
                    # Copy data to main buffer atomically
                    self.samples = local_samples.copy()
                    self.trigger_time = local_trigger_time
                    self.trigger_found = True
                    self.buffer_ready = True
                    self.status['samples_collected'] += sample_count
                    self.status['triggers_found'] += 1
                    self.status['last_trigger_time'] = local_trigger_time
                    
                    # Reset for next buffer
                    local_trigger_found = False
                    local_index = -1  # Will become 0 after increment
                    sample_count = 0
                    
                    print(f"Core 1: Buffer ready for processing")
                
                local_index = (local_index + 1) % Config.BUFFER_SIZE
                
                # Minimal delay to prevent overheating
                # On RP2040, we can run nearly at full speed
                if local_index % 1000 == 0:
                    utime.sleep_us(1)  # 1 microsecond pause every 1000 samples
                    
        except Exception as e:
            print(f"Core 1 error: {e}")
            self.core1_error = str(e)
            self.core1_running = False
        
        # Update final sample count when stopping
        self.status['samples_collected'] += loop_counter % 100000
        print("Core 1: Data collection stopped")
    
    def _upload_data(self, data_window):
        """Upload data with improved resource management and shorter timeouts"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": Config.AUTH_TOKEN,
                "User-Agent": "TRNG-Device/1.0"
            }
            
            response = urequests.post(
                f"https://{Config.HOST}/{Config.ENDPOINT}",
                json=data_window,
                headers=headers,
                timeout=2  # Reduced from 5 to 2 seconds to prevent blocking
            )
            
            if response.status_code == 200:
                print("Upload successful:", response.text[:50])  # Reduced log output
                response.close()  # Properly close response
                return True
            else:
                print(f"Upload failed with status: {response.status_code}")
                response.close()
                return False
                
        except Exception as e:
            error_msg = f"Upload error: {e}"
            print(error_msg)
            self.add_error(error_msg)
            return False
    
    def _send_ping(self):
        """Send ping to backend server with current timestamp"""
        try:
            # Get current time and format it
            current_time = utime.localtime()
            timestamp_str = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(
                current_time[0], current_time[1], current_time[2],
                current_time[3], current_time[4], current_time[5]
            )
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": Config.AUTH_TOKEN,
                "User-Agent": "TRNG-Device/1.0"
            }
            
            payload = {
                "timestamp": timestamp_str
            }
            
            response = urequests.get(
                f"https://{Config.HOST}/{Config.PING_ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=3
            )
            
            if response.status_code == 200:
                print(f"Ping successful: {timestamp_str}")
                response.close()
                self.status['ping_successful'] += 1
                self.status['last_ping_time'] = utime.time()
                return True
            else:
                print(f"Ping failed with status: {response.status_code}")
                response.close()
                self.status['ping_failed'] += 1
                return False
                
        except Exception as e:
            error_msg = f"Ping error: {e}"
            print(error_msg)
            self.add_error(error_msg)
            self.status['ping_failed'] += 1
            return False
    
    def w5x00_init(self):
        """Initialize W5x00 network interface with DHCP support"""
        try:
            spi = SPI(0, 2_000_000, mosi=Pin(19), miso=Pin(16), sck=Pin(18))
            nic = network.WIZNET5K(spi, Pin(17), Pin(20))
            nic.active(True)
            
            # Use DHCP to obtain IP address automatically
            print("Configuring network interface for DHCP...")
            nic.ifconfig('dhcp')
            
            attempts = 0
            while not nic.isconnected() and attempts < Config.MAX_CONNECTION_ATTEMPTS:
                print(f"Attempting DHCP connection: {attempts + 1}/{Config.MAX_CONNECTION_ATTEMPTS}")
                utime.sleep(2)  # Longer wait for DHCP negotiation
                attempts += 1
                self.wdt.feed()
            
            if not nic.isconnected():
                print("DHCP failed, falling back to static IP configuration...")
                # Fallback to static IP if DHCP fails
                nic.ifconfig((
                    Config.NETWORK_IP,
                    Config.NETWORK_MASK,
                    Config.NETWORK_GATEWAY,
                    Config.NETWORK_DNS
                ))
                
                # Wait for static connection
                attempts = 0
                while not nic.isconnected() and attempts < Config.MAX_CONNECTION_ATTEMPTS:
                    print(f"Attempting static connection: {attempts + 1}/{Config.MAX_CONNECTION_ATTEMPTS}")
                    utime.sleep(1)
                    attempts += 1
                    self.wdt.feed()
                
                if not nic.isconnected():
                    raise Exception("Failed to connect to the network with both DHCP and static IP")
            
            self.status['network_connected'] = True
            network_config = nic.ifconfig()
            print(f"Network connected successfully!")
            print(f"  IP Address: {network_config[0]}")
            print(f"  Subnet Mask: {network_config[1]}")
            print(f"  Gateway: {network_config[2]}")
            print(f"  DNS Server: {network_config[3]}")
            return nic
            
        except Exception as e:
            error_msg = f"Network initialization failed: {e}"
            print(error_msg)
            self.add_error(error_msg)
            raise
    
    def init_device(self):
        """Initialize device with comprehensive error handling"""
        print("Initializing device...")
        
        # Initialize network
        nic = self.w5x00_init()
        
        # Test connectivity
        try:
            uping.ping(Config.HOST, count=1, quiet=True)
            print("Host connectivity verified")
        except Exception as e:
            print(f"Warning: Host ping failed: {e}")
        
        # Synchronize time
        self._sync_ntp_time()
        
        print("Device initialization complete")
        return nic
    
    def _sync_ntp_time(self):
        """Synchronize NTP time with exponential backoff"""
        attempts = 0
        while attempts < Config.MAX_NTP_ATTEMPTS:
            self.wdt.feed()
            try:
                ntptime.host = "pool.ntp.org"
                ntptime.settime()
                current_time = utime.localtime()
                self.status['ntp_synchronized'] = True
                print(f"NTP time synchronized: {current_time}")
                return
            except Exception as e:
                print(f"Failed to synchronize NTP time (attempt {attempts + 1}): {e}")
                attempts += 1
                if attempts < Config.MAX_NTP_ATTEMPTS:
                    delay = Config.NTP_RETRY_DELAY_BASE ** attempts
                    print(f"Retrying in {delay} seconds...")
                    utime.sleep(delay)
        
        raise Exception("NTP time could not be synchronized after several attempts")
    
    def _handle_web_request(self, client_socket):
        """Handle individual web requests"""
        try:
            # Set client socket timeout to prevent hanging
            client_socket.settimeout(5.0)
            
            # Read request with timeout protection
            request = client_socket.recv(1024).decode('utf-8')
            
            # Parse request
            if not request:
                return
                
            lines = request.split('\n')
            if not lines:
                return
                
            request_line = lines[0].strip()
            if not request_line:
                return
                
            parts = request_line.split(' ')
            if len(parts) < 3:
                return
                
            method, path, _ = parts
            
            # Handle different endpoints
            if path == '/' or path == '/status':
                response = self._generate_status_page()
            elif path == '/api/status':
                response = self._generate_status_json()
            elif path == '/api/restart' and method == 'POST':
                response = self._handle_restart_request()
            elif path == '/api/config':
                response = self._generate_config_json()
            else:
                response = self._generate_404_page()
            
            # Send response in chunks to avoid timeout
            response_bytes = response.encode('utf-8')
            chunk_size = 1024
            for i in range(0, len(response_bytes), chunk_size):
                chunk = response_bytes[i:i + chunk_size]
                client_socket.send(chunk)
            
        except Exception as e:
            error_msg = f"Error handling web request: {e}"
            print(error_msg)
            self.add_error(error_msg)
            # Send a simple error response
            try:
                error_response = "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\nServer Error"
                client_socket.send(error_response.encode('utf-8'))
            except:
                pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _generate_status_page(self):
        """Generate compact HTML status page"""
        current_time = utime.time()
        uptime = current_time - self.status['uptime_start']
        
        # Convert uptime to readable format
        uptime_hours = int(uptime // 3600)
        uptime_minutes = int((uptime % 3600) // 60)
        uptime_seconds = int(uptime % 60)
        uptime_str = f"{uptime_hours}h {uptime_minutes}m {uptime_seconds}s"
        
        # Calculate sampling rate with proper check
        sampling_rate = 0.0
        if uptime > 0 and self.status['samples_collected'] > 0:
            sampling_rate = float(self.status['samples_collected']) / float(uptime)
        
        # Debug print for troubleshooting
        print(f"Web Debug: uptime={uptime:.1f}s, samples={self.status['samples_collected']}, rate={sampling_rate:.1f}Hz")
        
        # Get last error
        last_error = "None"
        if self.status['errors']:
            last_error = self.status['errors'][-1]['message'][:50] + "..." if len(self.status['errors'][-1]['message']) > 50 else self.status['errors'][-1]['message']
        
        # Core status
        core1_status = "Running" if self.core1_running else "🔴 Stopped"
        
        html = f"""HTTP/1.1 200 OK\r
Content-Type: text/html\r
Connection: close\r
\r
<!DOCTYPE html>
<html>
<head>
    <title>TRNG Status</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 15px; border-radius: 5px; }}
        .status {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }}
        .card {{ background: #f8f9fa; padding: 10px; border-radius: 3px; }}
        .value {{ font-size: 1.2em; font-weight: bold; color: #007bff; }}
        .error {{ background: #fff3cd; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .button {{ background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 3px; margin: 5px; cursor: pointer; }}
    </style>
    <script>
        function restart() {{
            if (confirm('Restart device?')) {{
                fetch('/api/restart', {{ method: 'POST' }});
                alert('Restart initiated');
            }}
        }}
        setTimeout(() => location.reload(), 15000);
    </script>
</head>
<body>
    <div class="container">
        <h2>TRNG Device - RP2040 Dual-Core</h2>
        
        <div class="status">
            <div class="card">
                <div class="value">{'Online' if self.status['network_connected'] else 'Offline'}</div>
                <small>Network</small>
            </div>
            <div class="card">
                <div class="value">{uptime_str}</div>
                <small>Uptime</small>
            </div>
            <div class="card">
                <div class="value">{self.status['samples_collected']:,}</div>
                <small>Samples</small>
            </div>
            <div class="card">
                <div class="value">{sampling_rate:.1f} Hz</div>
                <small>Sampling Rate</small>
            </div>
            <div class="card">
                <div class="value">{self.status['triggers_found']}</div>
                <small>Triggers</small>
            </div>
            <div class="card">
                <div class="value">{self.status['uploads_successful']}/{self.status['uploads_failed']}</div>
                <small>Uploads (OK/Fail)</small>
            </div>
            <div class="card">
                <div class="value">{self.status['ping_successful']}/{self.status['ping_failed']}</div>
                <small>Pings (OK/Fail)</small>
            </div>
            <div class="card">
                <div class="value">{self.status['current_queue_size']}</div>
                <small>Queue Size</small>
            </div>
            <div class="card">
                <div class="value">{core1_status}</div>
                <small>Core 1 Status</small>
            </div>
        </div>
        
        <div class="error">
            <strong>Last Error:</strong> {last_error}
        </div>
        
        <div style="text-align: center;">
            <button class="button" onclick="location.reload()">Refresh</button>
            <button class="button" onclick="restart()">Restart</button>
            <a href="/api/status" class="button">JSON</a>
        </div>
        
        <small>Auto-refresh: 15s | Core 0: System | Core 1: Data Collection</small>
    </div>
</body>
</html>"""
        return html
    
    def _generate_status_json(self):
        """Generate JSON status response"""
        current_time = utime.time()
        uptime = current_time - self.status['uptime_start']
        
        # Calculate sampling rate with proper checks
        sampling_rate = 0.0
        if uptime > 0 and self.status['samples_collected'] > 0:
            sampling_rate = float(self.status['samples_collected']) / float(uptime)
        
        status_data = {
            'device_info': {
                'device_id': 'trng_001',
                'firmware_version': '1.0',
                'uptime_seconds': uptime
            },
            'performance': {
                'samples_collected': self.status['samples_collected'],
                'sampling_rate_hz': sampling_rate,
                'triggers_found': self.status['triggers_found'],
                'uploads_successful': self.status['uploads_successful'],
                'uploads_failed': self.status['uploads_failed'],
                'ping_successful': self.status['ping_successful'],
                'ping_failed': self.status['ping_failed'],
                'queue_size': self.status['current_queue_size'],
                'queue_overflows': self.status['queue_overflows']
            },
            'system': {
                'network_connected': self.status['network_connected'],
                'ntp_synchronized': self.status['ntp_synchronized'],
                'memory_free_bytes': self.status['memory_free'],
                'threshold_voltage_mv': Config.THRESHOLD_VOLTAGE_MV,
                'running': self.running,
                'dual_core': {
                    'core0_operations': 'uploads, web_server, monitoring',
                    'core1_data_collection': self.core1_running,
                    'architecture': 'RP2040 ARM Cortex-M0+ dual-core'
                }
            },
            'timestamps': {
                'last_trigger': self.status['last_trigger_time'],
                'last_upload': self.status['last_upload_time'],
                'last_ping': self.status['last_ping_time'],
                'current_time': utime.time()
            },
            'errors': self.status['errors'][-5:]  # Last 5 errors
        }
        
        response = f"""HTTP/1.1 200 OK
Content-Type: application/json
Connection: close

{json.dumps(status_data)}"""
        return response
    
    def _generate_config_json(self):
        """Generate configuration JSON response"""
        config_data = {
            'network': {
                'host': Config.HOST,
                'ip': Config.NETWORK_IP,
                'gateway': Config.NETWORK_GATEWAY
            },
            'adc': {
                'pin': Config.ADC_PIN,
                'threshold_mv': Config.THRESHOLD_VOLTAGE_MV,
                'buffer_size': Config.BUFFER_SIZE,
                'reference_voltage': Config.ADC_REFERENCE_VOLTAGE
            },
            'timing': {
                'watchdog_timeout_ms': Config.WATCHDOG_TIMEOUT_MS,
                'upload_throttle_ms': Config.UPLOAD_THROTTLE_MS,
                'max_queue_size': Config.MAX_QUEUE_SIZE
            }
        }
        
        response = f"""HTTP/1.1 200 OK
Content-Type: application/json
Connection: close

{json.dumps(config_data)}"""
        return response
    
    def _handle_restart_request(self):
        """Handle device restart request"""
        print("Restart requested via web interface")
        self.restart_requested = True
        
        response_data = {
            'success': True,
            'message': 'Device restart initiated. Please wait...'
        }
        
        response = f"""HTTP/1.1 200 OK
Content-Type: application/json
Connection: close

{json.dumps(response_data)}"""
        return response
    
    def _generate_404_page(self):
        """Generate 404 error page"""
        html = """HTTP/1.1 404 Not Found
Content-Type: text/html
Connection: close

<!DOCTYPE html>
<html>
<head>
    <title>404 - Not Found</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <h1 class="error">404 - Page Not Found</h1>
    <p>The requested page could not be found.</p>
    <a href="/">Return to Status Page</a>
</body>
</html>"""
        return html
    
    def start(self):
        """Start the TRNG device operation - RP2040 dual-core approach"""
        try:
            self.init_device()
            
            print("Starting RP2040 dual-core operation:")
            print("  Core 0: Network, uploads, web server")
            print("  Core 1: Dedicated ADC data collection")
            
            # Start Core 1 for dedicated data collection
            print("Starting Core 1 data collection thread...")
            _thread.start_new_thread(self._core1_data_collection, ())
            
            # Wait for Core 1 to initialize
            utime.sleep(1)
            if not self.core1_running:
                raise Exception("Core 1 failed to start")
            
            # Initialize web server socket
            web_server_socket = None
            if Config.WEB_ENABLED:
                try:
                    web_server_socket = self._init_web_server_socket()
                    print("Web server socket initialized on Core 0")
                except Exception as e:
                    print(f"Warning: Could not initialize web server: {e}")
            
            # Start Core 0 operations (uploads, web server, etc.)
            self._run_core0_operations(web_server_socket)
            
        except KeyboardInterrupt:
            print("Shutting down...")
            self.running = False
            self.core1_running = False
        except Exception as e:
            print(f"Fatal error: {e}")
            # Ensure clean shutdown on error
            self.stop()
            raise
    
    def _init_web_server_socket(self):
        """Initialize web server socket without starting a thread"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((Config.WEB_HOST, Config.WEB_PORT))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Longer timeout for better stability
        print(f"Web interface available at http://{Config.NETWORK_IP}:{Config.WEB_PORT}")
        return server_socket
    
    def _run_core0_operations(self, web_server_socket):
        """Core 0: Handle uploads, web server, and system operations
        
        This function runs on the main CPU core and handles all
        non-critical operations that can tolerate some latency.
        """
        print("Core 0: Starting system operations")
        
        # Debugging counters
        loop_counter = 0
        
        # Upload variables - much more aggressive
        upload_counter = 0
        upload_interval = max(1, Config.UPLOAD_THROTTLE_MS // 10)  # Minimum 1, check every 10ms
        
        # Web server variables
        web_counter = 0
        web_interval = 20  # Check web server every 200ms (less frequent to give uploads priority)
        
        # Status update variables
        status_counter = 0
        status_interval = 400  # Update status every 4 seconds (watchdog is 8s)
        
        # Watchdog feeding - critical for stability
        watchdog_counter = 0
        watchdog_interval = 20  # Feed watchdog every 200ms (more frequent due to faster uploads)
        
        # Ping variables - send ping every 5 minutes
        ping_counter = 0
        ping_interval = Config.PING_INTERVAL_SECONDS * 100  # Convert to loop iterations (10ms each)
        last_ping_sent = utime.time()
        
        while self.running:
            try:
                loop_counter += 1
                
                # Debug output every 1000 loops
                if loop_counter % 1000 == 0:
                    print(f"Core 0: Loop {loop_counter}, Core1: {self.core1_running}")
                
                # Feed watchdog more frequently to prevent resets
                watchdog_counter += 1
                if watchdog_counter >= watchdog_interval:
                    self.wdt.feed()
                    watchdog_counter = 0
                
                # Check for errors from Core 1
                if self.core1_error:
                    error_msg = f"Core 1 error: {self.core1_error}"
                    print(error_msg)
                    self.add_error(error_msg)
                    self.core1_error = None
                
                # Check if Core 1 is still running
                if not self.core1_running and self.running:
                    error_msg = "Core 1 stopped unexpectedly"
                    print(error_msg)
                    self.add_error(error_msg)
                    break
                
                # 1. Process completed data buffers from Core 1
                if self.buffer_ready:
                    self._process_completed_buffer()
                
                # 2. Data Upload (more aggressive processing)
                upload_counter += 1
                if upload_counter >= upload_interval and self.data_queue:
                    # Process multiple uploads per cycle if queue is getting full
                    uploads_this_cycle = min(2, len(self.data_queue))  # Reduced to 2 uploads per cycle
                    if len(self.data_queue) > Config.MAX_QUEUE_SIZE * 0.6:  # If queue is 60% full
                        uploads_this_cycle = min(3, len(self.data_queue))  # Process up to 3
                    
                    successful_uploads = 0
                    for _ in range(uploads_this_cycle):
                        if not self.data_queue:
                            break
                            
                        data_window = self.data_queue.pop(0)
                        
                        # Feed watchdog before potentially blocking upload
                        self.wdt.feed()
                        
                        success = self._upload_data(data_window)
                        if success:
                            self.status['uploads_successful'] += 1
                            self.status['last_upload_time'] = utime.time()
                            successful_uploads += 1
                        else:
                            self.status['uploads_failed'] += 1
                            # Re-queue failed upload for retry (but limit retries)
                            """ if len(self.data_queue) < Config.MAX_QUEUE_SIZE:
                                if not hasattr(data_window, 'retry_count'):
                                    data_window['retry_count'] = 1
                                    self.data_queue.insert(0, data_window)
                                elif data_window.get('retry_count', 0) < 2:  # Reduced retries to 2
                                    data_window['retry_count'] = data_window.get('retry_count', 0) + 1
                                    self.data_queue.insert(0, data_window)
                                # After 2 retries, drop the data to prevent queue overflow """
                            break  # Stop processing more uploads on failure to prevent cascade failures
                    
                    # Debug output for upload performance
                    if successful_uploads > 0:
                        print(f"Core 0: {successful_uploads} uploads completed, queue: {len(self.data_queue)}")
                    
                    self.status['current_queue_size'] = len(self.data_queue)
                    upload_counter = 0
                
                # 3. Web Server (periodically)
                if web_server_socket:
                    web_counter += 1
                    if web_counter >= web_interval:
                        try:
                            client_socket, addr = web_server_socket.accept()
                            print(f"Web client connected from {addr}")
                            # Handle request in a separate function to avoid blocking
                            self._handle_web_request(client_socket)
                        except OSError as e:
                            # Normal timeout, no connection available
                            if e.errno != 110:  # ETIMEDOUT
                                print(f"Web server socket error: {e}")
                        web_counter = 0
                
                # 4. Ping to backend server (every 5 minutes)
                ping_counter += 1
                if ping_counter >= ping_interval:
                    current_time = utime.time()
                    time_since_last_ping = current_time - last_ping_sent
                    
                    # Double-check timing to ensure we don't ping too frequently
                    if time_since_last_ping >= Config.PING_INTERVAL_SECONDS:
                        print(f"Core 0: Sending ping to backend (last ping: {time_since_last_ping:.0f}s ago)")
                        self.wdt.feed()  # Feed watchdog before network operation
                        self._send_ping()
                        last_ping_sent = current_time
                    
                    ping_counter = 0
                
                # 5. Status updates and maintenance
                status_counter += 1
                if status_counter >= status_interval:
                    self.update_memory_status()
                    status_counter = 0
                    
                    if self.restart_requested:
                        print("Restart requested, shutting down...")
                        self.running = False
                        self.core1_running = False
                        break
                
                # Core 0 can afford longer delays since it's not time-critical
                utime.sleep_ms(10)
                
            except Exception as e:
                error_msg = f"Error in Core 0 operations: {e}"
                print(error_msg)
                self.add_error(error_msg)
                utime.sleep_ms(100)  # Longer delay on error
        
        # Clean up web server socket
        if web_server_socket:
            try:
                web_server_socket.close()
            except:
                pass
            print("Core 0: Web server socket closed")
        
        print("Core 0: System operations stopped")
    
    def _process_completed_buffer(self):
        """Process a completed data buffer from Core 1"""
        try:
            # Create data window from the completed buffer
            data_window = {
                "timestamp": self.trigger_time,
                "data": self.samples.copy(),
                "device_id": "trng_001",
                "buffer_size": Config.BUFFER_SIZE
            }
            
            # Add to queue with better overflow handling
            if len(self.data_queue) < Config.MAX_QUEUE_SIZE:
                self.data_queue.append(data_window)
                self.status['current_queue_size'] = len(self.data_queue)
                print(f"Core 0: Data queued. Queue size: {len(self.data_queue)}/{Config.MAX_QUEUE_SIZE}")
            else:
                print(f"Core 0: Queue full ({len(self.data_queue)}/{Config.MAX_QUEUE_SIZE}), dropping oldest data")
                # Drop oldest data
                dropped_data = self.data_queue.pop(0)
                self.data_queue.append(data_window)
                self.status['queue_overflows'] += 1
                self.status['current_queue_size'] = len(self.data_queue)
                print(f"Core 0: Dropped data from timestamp {dropped_data.get('timestamp', 'unknown')}")
            
            # Reset buffer ready flag
            self.buffer_ready = False
            
            # Trigger garbage collection
            gc.collect()
            
        except Exception as e:
            error_msg = f"Error processing completed buffer: {e}"
            print(error_msg)
            self.add_error(error_msg)
    
    def restart_device(self):
        """Restart the entire device"""
        print("Restarting device...")
        self.running = False
        utime.sleep(2)  # Give threads time to stop
        machine.reset()  # Hardware reset

def main():
    """Main entry point with restart loop"""
    print("TRNG Device Starting...")
    print("Platform: RP2040 Dual-Core with W5500-EVB-Pico")
    print("Architecture: Core 0 (System) + Core 1 (Data Collection)")
    print("Watchdog timeout: 8 seconds (RP2040 maximum)")
    
    device = None
    startup_attempt = 0
    
    while True:
        try:
            startup_attempt += 1
            print(f"Startup attempt #{startup_attempt}")
            
            # Clean up previous device instance if it exists
            if device:
                print("Cleaning up previous device instance...")
                device.stop()
                device = None
                # Extra delay to ensure threads are fully stopped
                utime.sleep(3)  # Longer delay for RP2040
                gc.collect()
                print("Cleanup completed")
            
            print("Creating new device instance...")
            device = TRNGDevice()
            
            print("Starting device...")
            device.start()
            
            # Check if restart was requested
            if device.restart_requested:
                print("Restarting device as requested...")
                device.stop()
                utime.sleep(3)  # Give threads time to stop
                machine.reset()  # Hardware reset
            else:
                break  # Normal exit
                
        except Exception as e:
            print(f"Device error in attempt #{startup_attempt}: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Print traceback for MicroPython
            try:
                import traceback
                traceback.print_exc()
            except:
                print("Could not print traceback")
            
            if device:
                print("Stopping device after error...")
                device.stop()
            
            # Exponential backoff for restart delay
            restart_delay = min(10 + (startup_attempt * 2), 60)  # Max 60 seconds
            print(f"Restarting in {restart_delay} seconds...")
            utime.sleep(restart_delay)
            gc.collect()  # Clean up memory before retry

if __name__ == "__main__":
    main()
