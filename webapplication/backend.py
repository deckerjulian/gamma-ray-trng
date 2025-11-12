import os
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from datetime import datetime
import logging
from random import choice
from random_username.generate import generate_username
from datetime import datetime, timedelta
import threading
import time
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import time
from html import escape
from better_profanity import profanity
import eventlet
from nist_randomness_tests import test_random_bits_file
from dotenv import load_dotenv

matplotlib.use("Agg")

# Ensure we have absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Load environment variables from .env file
load_dotenv(os.path.join(INSTANCE_DIR, ".env"))

# Random bits file with absolute path
file_random_bits = os.path.join(INSTANCE_DIR, "random_bits.txt")

# map IP addresses to their last message's timestamp
last_message_time_by_ip = {}

# last ping time of device
last_ping_time = "Never"

# Use absolute path for static folder
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static",
    instance_path=INSTANCE_DIR,
)


# Use absolute path for database
db_path = os.path.join(INSTANCE_DIR, "data.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Load CORS allowed origins from environment variable
cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5000").split(",")
socketio = SocketIO(app, cors_allowed_origins=cors_origins)


# Model definition
class GammaEventEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    data = db.Column(db.JSON, nullable=False)
    image_plot = db.Column(db.Text)
    used_for_answer = db.Column(db.Boolean, default=False)  # New flag


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String, nullable=False)
    answer = db.Column(db.String, default="")
    answer_image_plot = db.Column(db.Text)
    username = db.Column(db.String, default="")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Load secret token from environment variable
SECRET_TOKEN = os.getenv("SECRET_TOKEN")


def create_and_save_plot(data_dict):
    plt.axis("off")
    plt.plot(data_dict["data"], color="black")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return image_base64


def check_random_bits():
    """
    Check if the number of GammaEventEntries is dividible by 4 and calculate a random bit.
    If the number of entries is dividible by 4, calculate a random bit based on
    the last 4 entries and save it to a file.
    """
    random_bit = calculate_random_bit_from_gamma_events()
    if random_bit is not None:
        with open(file_random_bits, "a") as f:
            f.write(str(random_bit) + "\n")
        print(f"Random bit calculated and saved: {random_bit}")


def calculate_random_bit_from_gamma_events():
    """
    Calculate a random bit based on the last 4 GammaEventEntries.
    The bit is determined by the time difference between the first two and the last two entries.
    If the first time difference is greater than or equal to the second, return 1,
    otherwise return 0.
    """
    # Check if the number of GammaEventEntries is dividible by 4
    count = GammaEventEntry.query.count()
    if count % 4 != 0:
        return None
    # Select the last 4 entries
    entries = (
        GammaEventEntry.query.order_by(GammaEventEntry.timestamp.desc()).limit(4).all()
    )
    if len(entries) < 4:
        return None
    # Calculate the time differences
    time_values = [entry.timestamp for entry in entries]
    time_diff1 = time.mktime(time_values[1].timetuple()) - time.mktime(
        time_values[0].timetuple()
    )
    time_diff2 = time.mktime(time_values[3].timetuple()) - time.mktime(
        time_values[2].timetuple()
    )
    time_diff_res = time_diff1 - time_diff2
    # Determine the bit based on the time difference
    if time_diff_res >= 0:
        return 1
    else:
        return 0


def run_nist_randomness_tests():
    """
    Run NIST randomness tests on the random bits file.
    This function should be called periodically to verify the quality of randomness.

    Returns:
        dict: Test results from NIST test suite
    """
    try:
        print("Running NIST randomness tests...")
        results = test_random_bits_file(file_random_bits)

        # Log summary
        if results:
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r["passed"])
            success_rate = passed_tests / total_tests

            print(
                f"NIST Test Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})"
            )

            # Log individual test results
            for test_name, result in results.items():
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  {test_name}: {status} (p-value: {result['p_value']:.6f})")

            return results
        else:
            print("No test results obtained - insufficient data or file not found")
            return {}

    except Exception as e:
        print(f"Error running NIST tests: {e}")
        return {}


def check_for_answers():
    empty_answer_messages = ChatMessage.query.filter(ChatMessage.answer == "")
    for message in empty_answer_messages:
        GammaEventEntries = (
            GammaEventEntry.query.filter(
                GammaEventEntry.timestamp > message.timestamp,
                GammaEventEntry.used_for_answer == False,  # Only select unused entries
            )
            .limit(4)
            .all()
        )

        if len(GammaEventEntries) >= 4:
            print("The answer is ready")
            answer = ""
            timeValues = [entry.timestamp for entry in GammaEventEntries]
            timeDiff1 = time.mktime(timeValues[1].timetuple()) - time.mktime(
                timeValues[0].timetuple()
            )
            timeDiff2 = time.mktime(timeValues[3].timetuple()) - time.mktime(
                timeValues[2].timetuple()
            )
            timeDiffRes = timeDiff1 - timeDiff2
            answer = "Yeah!" if timeDiffRes >= 0 else "Nope..."

            images = []
            for entry in GammaEventEntries:
                if entry.image_plot:
                    image_data = base64.b64decode(entry.image_plot)
                    image = Image.open(io.BytesIO(image_data))
                    images.append(image)
                entry.used_for_answer = True  # Mark as used

            db.session.commit()  # Commit marking as used

            if images:
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                combined_image = Image.new("RGBA", (total_width, max_height))
                x_offset = 0
                for img in images:
                    img = img.convert("RGBA")
                    combined_image.paste(img, (x_offset, 0), img)
                    x_offset += img.width

                buffered = io.BytesIO()
                combined_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

            message.answer = answer
            message.answer_image_plot = img_str
            db.session.commit()

            with app.app_context():
                socketio.emit(
                    "receive_message",
                    {
                        "id": message.id,
                        "message": message.question,
                        "username": message.username,
                        "answer": message.answer,
                        "answer_image_plot": message.answer_image_plot,
                        "timestamp": message.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                
@app.route("/getlastping", methods=["GET"])
def get_last_ping():
    global last_ping_time
    if last_ping_time == "Never":
        return jsonify({"last_ping": last_ping_time}), 200
    if isinstance(last_ping_time, str):
        return jsonify({"last_ping": last_ping_time}), 200
    
    return jsonify({"last_ping": last_ping_time.strftime("%Y-%m-%d %H:%M:%S")}), 200

@app.route("/ping", methods=["GET"])
def ping():
    global last_ping_time
    token = request.headers.get("Authorization")
    
    if token != SECRET_TOKEN:
        logging.warning("Unauthorized access attempt.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    # Check if there is a json attached to the request
    if request.is_json:
        data = request.get_json()
        if "timestamp" in data:
            logging.info(f"Ping received with timestamp: {data['timestamp']}")
            
            # Update the last ping time
            last_ping_time = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
            current_server_time = datetime.utcnow()
            
            # Calculate the time difference
            time_diff = (current_server_time - last_ping_time).total_seconds()
            logging.info(f"Time difference from ping: {time_diff} seconds")
            
        else:
            logging.info("Ping received without timestamp")
            last_ping_time = datetime.utcnow()

    return jsonify({"status": "success", "message": "Pong!"}), 200

@app.route("/upload", methods=["POST"])
def upload_data():
    token = request.headers.get("Authorization")

    if token != SECRET_TOKEN:
        logging.warning("Unauthorized access attempt.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if request.is_json:
        data = request.get_json()
        new_entry = GammaEventEntry(data=data)
        # Create plot and store it as a binary blob
        new_entry.image_plot = create_and_save_plot(data)

        db.session.add(new_entry)
        db.session.commit()

        # Emit new data to all connected clients
        emit_data = {
            "id": new_entry.id,
            "timestamp": new_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "data": new_entry.data,
            "image_plot": new_entry.image_plot,
        }
        socketio.emit("new_data", emit_data)

        check_random_bits()
        check_for_answers()

        return jsonify({"status": "success", "message": "Data received"}), 200
    else:
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400


@app.route("/")
def index():
    try:
        file_path = os.path.join(app.static_folder, "index.html")
        if not os.path.exists(file_path):
            logging.error(f"index.html nicht gefunden unter: {file_path}")
            return "index.html nicht gefunden", 404
        if not os.access(file_path, os.R_OK):
            logging.error(f"Keine Leserechte für index.html unter: {file_path}")
            return "Keine Leserechte für index.html", 403
        return send_from_directory(app.static_folder, "index.html")
    except Exception as e:
        logging.error(f"Fehler beim Ausliefern von index.html: {e}")
        return f"Fehler beim Ausliefern von index.html: {e}", 500


@app.route("/health")
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({"status": "healthy"}), 200


@app.route("/nist_tests", methods=["GET"])
def run_nist_tests_endpoint():
    """
    API endpoint to run NIST randomness tests on the random bits file.
    Returns JSON with test results and file statistics.
    """
    try:
        # Get file statistics first
        file_stats = get_random_bits_statistics()

        # Only run tests if we have sufficient data
        if file_stats["bit_count"] < 1000:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Insufficient data for NIST tests. Need at least 1000 bits, but only have {file_stats['bit_count']} bits.",
                        "file_stats": file_stats,
                        "results": {},
                    }
                ),
                400,
            )

        results = run_nist_randomness_tests()

        if not results:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "No test results - error during test execution",
                        "file_stats": file_stats,
                        "results": {},
                    }
                ),
                400,
            )

        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if bool(r["passed"]))
        success_rate = passed_tests / total_tests

        # Format results for JSON response - convert numpy types to Python native types
        formatted_results = {}
        for test_name, result in results.items():
            formatted_results[test_name] = {
                "passed": bool(result["passed"]),  # Convert numpy bool_ to Python bool
                "p_value": float(
                    round(result["p_value"], 6)
                ),  # Ensure it's a Python float
                "status": "PASS" if bool(result["passed"]) else "FAIL",
            }

        return (
            jsonify(
                {
                    "status": "success",
                    "file_stats": file_stats,
                    "summary": {
                        "total_tests": int(total_tests),  # Ensure it's a Python int
                        "passed_tests": int(passed_tests),  # Ensure it's a Python int
                        "success_rate": float(
                            round(success_rate, 3)
                        ),  # Ensure it's a Python float
                        "overall_result": "PASS" if success_rate >= 0.8 else "FAIL",
                    },
                    "results": formatted_results,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Error in NIST tests endpoint: {e}")  # Add logging
        # Try to get file stats even if tests failed
        try:
            file_stats = get_random_bits_statistics()
        except:
            file_stats = {"error": "Could not read file statistics"}

        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error running NIST tests: {str(e)}",
                    "file_stats": file_stats,
                    "results": {},
                }
            ),
            500,
        )


def get_random_bits_statistics():
    """
    Get statistics about the random bits file.
    Returns dictionary with file information.
    """
    try:
        if not os.path.exists(file_random_bits):
            return {
                "file_exists": False,
                "bit_count": 0,
                "ones_count": 0,
                "zeros_count": 0,
                "file_size_bytes": 0,
                "error": "File does not exist",
            }

        # Get file size
        file_size = os.path.getsize(file_random_bits)

        # Read and analyze the file content
        with open(file_random_bits, "r") as f:
            content = f.read().strip()

        # Remove newlines and count bits
        bits_only = content.replace("\n", "").replace("\r", "")
        bit_count = len(bits_only)
        ones_count = bits_only.count("1")
        zeros_count = bits_only.count("0")

        # Calculate percentages
        ones_percentage = (ones_count / bit_count * 100) if bit_count > 0 else 0
        zeros_percentage = (zeros_count / bit_count * 100) if bit_count > 0 else 0

        return {
            "file_exists": True,
            "bit_count": bit_count,
            "ones_count": ones_count,
            "zeros_count": zeros_count,
            "ones_percentage": round(ones_percentage, 2),
            "zeros_percentage": round(zeros_percentage, 2),
            "file_size_bytes": file_size,
            "is_sufficient_for_nist": bit_count >= 1000,
        }

    except Exception as e:
        return {
            "file_exists": False,
            "bit_count": 0,
            "ones_count": 0,
            "zeros_count": 0,
            "file_size_bytes": 0,
            "error": f"Error reading file: {str(e)}",
        }


@socketio.on("connect")
def handle_connect():
    initial_data = (
        GammaEventEntry.query.order_by(GammaEventEntry.timestamp.desc()).limit(40).all()
    )
    data_list = [
        {
            "id": data.id,
            "timestamp": data.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "data": data.data,
            "image_plot": data.image_plot,
        }
        for data in initial_data
    ]
    emit("initial_data", data_list)


"""
    Chat System
"""


@socketio.on("send_message")
def handle_message(data):
    current_time = datetime.utcnow()
    ip_address = request.remote_addr

    # Retrieve and sanitize the message
    message_content = escape(data)  # Escapes HTML special characters

    if len(message_content) == 0:
        emit(
            "message_error",
            {"error": "Empty message is not allowed."},
            to=request.sid,
        )
        return

    # Character limit check
    if len(message_content) > 1000:
        emit(
            "message_error",
            {"error": "Message too long. Limit is 1000 characters."},
            to=request.sid,
        )
        return

    # Profanity and code injection checks
    if profanity.contains_profanity(message_content):
        emit(
            "message_error",
            {"error": "Message contains inappropriate content."},
            to=request.sid,
        )
        return

    # Rate limiting check
    if ip_address in last_message_time_by_ip and (
        current_time - last_message_time_by_ip[ip_address]
    ) < timedelta(seconds=30):
        emit(
            "message_error",
            {"error": "You can only send a message every 30 seconds."},
            to=request.sid,
        )
        return

    last_message_time_by_ip[ip_address] = current_time

    new_message = ChatMessage(
        question=message_content, username=generate_username(1)[0]
    )
    db.session.add(new_message)
    db.session.commit()

    # Broadcast server response
    socketio.emit(
        "message_added",
        {
            "id": new_message.id,
            "message": new_message.question,
            "username": new_message.username,
            "answer": new_message.answer,
            "answer_image_plot": new_message.answer_image_plot,
            "timestamp": new_message.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        },
        to=request.sid,
    )


@app.route("/latest_messages", methods=["GET"])
def get_latest_messages():
    messages = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).limit(100).all()
    messages = [
        {
            "id": msg.id,
            "message": msg.question,
            "username": msg.username,
            "answer": msg.answer,
            "answer_image_plot": msg.answer_image_plot,
            "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for msg in messages
    ]
    return jsonify(messages)


@app.route("/older_messages", methods=["GET"])
def get_older_messages():
    last_id = request.args.get(
        "last_id", type=int
    )  # ID of the oldest message currently displayed
    if last_id:
        messages = (
            ChatMessage.query.filter(ChatMessage.id < last_id)
            .order_by(ChatMessage.timestamp.asc())
            .limit(20)
            .all()
        )
        messages = [
            {
                "id": msg.id,
                "message": msg.question,
                "username": msg.username,
                "answer": msg.answer,
                "answer_image_plot": msg.answer_image_plot,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for msg in reversed(messages)
        ]
        return jsonify(messages)
    return jsonify([])


def cleanup_ip_timestamps():
    nist_test_counter = 0  # Counter to run NIST tests every 10 iterations

    while True:
        current_time = datetime.utcnow()
        with app.app_context():  # Ensure database access within app context
            with threading.Lock():
                # Clean up IP timestamps
                keys_to_remove = [
                    ip
                    for ip, last_time in last_message_time_by_ip.items()
                    if (current_time - last_time) > timedelta(seconds=30)
                ]
                for ip in keys_to_remove:
                    del last_message_time_by_ip[ip]

                # Count the number of GammaEventEntry and ChatMessage entries
                gamma_event_count = GammaEventEntry.query.count()
                chat_message_count = ChatMessage.query.count()

                print(f"GammaEventEntry count: {gamma_event_count}")
                print(f"ChatMessage count: {chat_message_count}")

                # Check and clean up GammaEventEntry table, keeping at least 30 unused entries
                unused_entries_count = GammaEventEntry.query.filter_by(
                    used_for_answer=False
                ).count()
                if unused_entries_count > 30:
                    ids_to_delete = (
                        db.session.query(GammaEventEntry.id)
                        .filter(GammaEventEntry.used_for_answer == False)
                        .order_by(GammaEventEntry.timestamp.asc())
                        .limit(unused_entries_count - 30)
                        .all()
                    )
                    ids_to_delete = [id[0] for id in ids_to_delete]
                    if ids_to_delete:
                        db.session.query(GammaEventEntry).filter(
                            GammaEventEntry.id.in_(ids_to_delete)
                        ).delete(synchronize_session=False)
                        db.session.commit()

                # Run NIST tests periodically (every 10 iterations = ~100 seconds)
                nist_test_counter += 1
                if nist_test_counter >= 10:
                    nist_test_counter = 0
                    try:
                        # Check if we have enough bits for testing (minimum 1000 bits recommended)
                        try:
                            with open(file_random_bits, "r") as f:
                                bits_content = f.read().strip().replace("\n", "")
                                if len(bits_content) >= 1000:
                                    print("Running periodic NIST randomness tests...")
                                    run_nist_randomness_tests()
                                else:
                                    print(
                                        f"Not enough bits for NIST tests: {len(bits_content)} (need at least 1000)"
                                    )
                        except FileNotFoundError:
                            print("Random bits file not found for NIST testing")
                    except Exception as e:
                        print(f"Error during periodic NIST testing: {e}")

        # Sleep for a specific interval before the next iteration
        time.sleep(10)


# Start the cleanup task as a greenlet
eventlet.spawn(cleanup_ip_timestamps)


# Initialize database and create tables
def init_database():
    """Initialize database and create all tables if they don't exist"""
    import os

    # Ensure instance directory exists with proper permissions
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    
    # Try to create a test file to verify write permissions
    try:
        test_file = os.path.join(INSTANCE_DIR, "test_write_permissions.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("Instance directory has write permissions")
    except Exception as e:
        print(f"Warning: Instance directory might not have write permissions: {e}")

    # Create all tables
    with app.app_context():
        try:
            db.create_all()
            print("Database initialized and tables created")
            
            # Test database write access
            test_count = GammaEventEntry.query.count()
            print(f"Database read test successful. Current entries: {test_count}")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise


# Flag to track if database is initialized
_db_initialized = False


def ensure_database_initialized():
    """Ensure database is initialized before first use"""
    global _db_initialized
    if not _db_initialized:
        init_database()
        _db_initialized = True


# Add database initialization to routes that need it
@app.before_request
def before_request():
    ensure_database_initialized()


if __name__ == "__main__":
    # For development, initialize immediately
    init_database()
