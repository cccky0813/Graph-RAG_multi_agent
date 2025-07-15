import os
import json
import io
import threading
import queue # For thread-safe communication
import tempfile
import uuid
from flask import Flask, render_template, request, session, redirect, url_for, Response, stream_with_context, jsonify, send_from_directory
import q_a # Assuming q_a.py is in the same directory or accessible via PYTHONPATH
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from ansi2html import Ansi2HTMLConverter # For converting rich's ANSI output to HTML
from py2neo import Graph as Py2neoGraph # Explicit import for clarity
from voice_asr import recognize_voice # Import our voice recognition module
from image_segmentation import image_segmentation_service # Import image segmentation service
from image_description import image_description_service # Import image description service

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session management

# Store the original print and console from q_a module to restore later
original_q_a_console_print = None
original_q_a_console_file = None
# This global variable will be set before calling RAG system
# to allow sse_print to yield data for the current request context.
# This is a workaround for monkey-patching in a web context.
# A more robust solution might involve passing a callback or using a queue.
current_sse_yield_callback = None


# --- Helper for Log Streaming ---
def sse_log_print(*args, **kwargs):
    """
    Monkey-patched print function for q_a.console.
    Captures rich output and yields it for SSE.
    """
    global current_sse_yield_callback
    if not current_sse_yield_callback:
        if original_q_a_console_print: # Fallback to original if no SSE context
            original_q_a_console_print(*args, **kwargs)
        return

    # Capture output to a string buffer
    s_io = io.StringIO()
    # Use a temporary console to capture the output of the rich object
    # Important: force_terminal=False and color_system=None for clean HTML export
    # However, for direct ANSI to HTML, we might need terminal=True
    
    # Heuristic: if the first arg is a rich renderable like Panel, Table, Markdown
    # try to export it as HTML directly.
    if args and hasattr(args[0], '__rich_console__'):
        is_complex_rich_obj = isinstance(args[0], (Panel, Table, Markdown))
        # Check if it's simple text or a rich renderable object.
        # If it's just a string with rich tags, Console(file=s_io).print will handle it.
        temp_console_for_export = Console(file=s_io, record=True, width=100, force_terminal=False, color_system=None)
        temp_console_for_export.print(*args, **kwargs)
        # Now get the HTML
        # export_html() is good for full documents. For snippets, we might need another way.
        # For now, let's try exporting the captured segment.
        # The `export_html` on a recorded console is more reliable.
        # Resetting s_io for this purpose.
        s_io.seek(0)
        s_io.truncate(0)
        recorded_console = Console(file=s_io, record=True, width=100) # Keep width reasonable
        recorded_console.print(*args, **kwargs)
        html_content = recorded_console.export_html(inline_styles=True, code_format="<pre class=\"code\">{code}</pre>")
        
        # Clean up the exported HTML a bit (remove doctype, html, body tags for partial content)
        if "<!DOCTYPE html>" in html_content:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            body_content = soup.body.decode_contents() if soup.body else html_content
            # Remove default white background from rich's body style if present
            body_content = body_content.replace("background-color:#ffffff;", "", 1).replace("color:#000000;", "", 1)
            current_sse_yield_callback(f"data: {json.dumps({'type': 'log_html', 'content': body_content})}\n\n")

    else: # Simple string or other arguments
        s_io = io.StringIO()
        # Create a console that writes to the string buffer, trying to preserve rich formatting
        # for ansi2html conversion.
        temp_console_for_ansi = Console(file=s_io, force_terminal=True, color_system="truecolor", width=100)
        temp_console_for_ansi.print(*args, **kwargs)
        ansi_output = s_io.getvalue()
        
        conv = Ansi2HTMLConverter(inline=True, scheme="solarized", linkify=False, dark_bg=True)
        html_output = conv.convert(ansi_output, full=False)
        #current_sse_yield_callback(f"data: {json.dumps({'type': 'log_html', 'content': f'<div class=\"log-entry-raw\">{html_output}</div>'})}\n\n")
        log_content = f'<div class="log-entry-raw">{html_output}</div>'
        current_sse_yield_callback(f"data: {json.dumps({'type': 'log_html', 'content': log_content})}\n\n")
    if original_q_a_console_print: # Optionally, also print to server console
        original_q_a_console_print(*args, **kwargs)

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    if "neo4j_config" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", neo4j_config=session["neo4j_config"])

@app.route("/login", methods=["GET", "POST"])
def login():
    default_config = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "123456789")
    }
    # q_a.py might have its own argument parsing for these, but Flask UI takes precedence here.

    if request.method == "POST":
        session["neo4j_config"] = {
            "uri": request.form.get("uri", default_config["uri"]),
            "user": request.form.get("user", default_config["user"]),
            "password": request.form.get("password", default_config["password"])
        }
        try:
            graph_check = Py2neoGraph(session["neo4j_config"]["uri"],
                                auth=(session["neo4j_config"]["user"], session["neo4j_config"]["password"]))
            graph_check.run("RETURN 1")
            session["neo4j_connected"] = True
            return redirect(url_for("index"))
        except Exception as e:
            session.pop("neo4j_config", None)
            session["neo4j_connected"] = False
            app.logger.error(f"Neo4j connection failed: {e}")
            return render_template("login.html", error=f"Connection failed: {e}", defaults=default_config)
    return render_template("login.html", defaults=default_config)


@app.route("/ask", methods=["POST"])
def ask_question():
    if "neo4j_config" not in session or not session.get("neo4j_connected"):
        return Response(json.dumps({"error": "Not connected to Neo4j. Please login."}), status=403, mimetype='application/json')

    data = request.json
    question_text = data.get("question")
    enable_multi_hop = data.get("enable_multi_hop", True)
    search_budget = data.get("search_budget", "Deeper")

    if not question_text:
        return Response(json.dumps({"error": "No question provided."}), status=400, mimetype='application/json')

    def generate_response_stream():
        message_queue = queue.Queue()
        # Flag to signal the main loop that the worker is done
        finished_signal = threading.Event() 

        class SseLogStreamWrapper(io.TextIOBase):
            def __init__(self, q):
                self.queue = q
                self.buffer = ""
                self.ansi_conv = Ansi2HTMLConverter(inline=True, scheme="solarized", linkify=False, dark_bg=True)
                # Ensure encoding issues don't arise if system default isn't UTF-8

            def write(self, s: str):
                if not isinstance(s, str):
                    try:
                        # Try decoding if it's bytes, assume utf-8 or default
                        s = s.decode(errors='replace')
                    except (AttributeError, UnicodeDecodeError):
                         s = str(s) # Fallback to string representation

                self.buffer += s
                # Process lines more carefully
                while True:
                    # Find the first newline character
                    try: newline_index = self.buffer.index('\n')
                    except ValueError: break # No complete line found yet

                    line_to_process = self.buffer[:newline_index + 1]
                    self.buffer = self.buffer[newline_index + 1:]

                    if line_to_process.strip():
                        # Convert only the line content, trim whitespace/newlines before conversion
                        html_line = self.ansi_conv.convert(line_to_process.strip(), full=False)
                        # Send HTML content wrapped in a div for styling
                        self.queue.put({'type': 'log_html', 'content': f"<div class='log-item'>{html_line}</div>"})

                return len(s.encode()) # Return bytes written

            def flush(self):
                # Process any remaining data in the buffer when flushed
                if self.buffer.strip():
                    html_line = self.ansi_conv.convert(self.buffer.strip(), full=False)
                    self.queue.put({'type': 'log_html', 'content': f"<div class='log-item'>{html_line}</div>"})
                    self.buffer = ""

            def isatty(self):
                return False # Important for rich

            # Ensure TextIOBase required methods are present if needed (read etc not needed for write-only)
            def readable(self): return False
            def seekable(self): return False
            def writable(self): return True


        def rag_worker(q, req_session, question, multi_hop, budget, finish_event):
            original_q_a_console_file = None
            worker_sse_wrapper = SseLogStreamWrapper(q) # Create wrapper instance for this thread

            try:
                # Store original global console file
                if hasattr(q_a, 'console') and hasattr(q_a.console, 'file'):
                    original_q_a_console_file = q_a.console.file
                
                # Patch global console file
                if hasattr(q_a, 'console'):
                    q_a.console.file = worker_sse_wrapper # Use the instance created above
                else:
                    q.put({'type': 'error', 'content': 'Internal error: q_a.console not found.'})
                    finish_event.set() # Signal finish even on error
                    return

                rag_system = q_a.Neo4jRAGSystem(
                    neo4j_uri=req_session["neo4j_config"]["uri"],
                    neo4j_user=req_session["neo4j_config"]["user"],
                    neo4j_password=req_session["neo4j_config"]["password"],
                    enable_multi_hop=multi_hop,
                    search_budget_mode=budget
                )

                # --- Blocking call ---
                final_answer = rag_system.answer_question(question)
                # --- End call ---
                
                # Ensure final flush of the wrapper *before* sending answer/finished
                worker_sse_wrapper.flush() 

                q.put({'type': 'answer', 'content': final_answer})

            except Exception as e:
                app.logger.error(f"Error in RAG worker thread: {e}", exc_info=True)
                # Ensure flush even if error happened during processing
                try: worker_sse_wrapper.flush()
                except: pass # Avoid errors during error handling
                q.put({'type': 'error', 'content': f"An error occurred: {str(e)}"})
            finally:
                 # Restore original console file **always**
                if original_q_a_console_file is not None and hasattr(q_a, 'console'):
                    q_a.console.file = original_q_a_console_file
                
                # Signal that the worker has finished processing
                finish_event.set() # Signal main loop via event
                q.put({'type': 'finished'}) # Also put signal in queue

        # Start the worker thread
        thread_session_data = dict(session)
        worker_thread = threading.Thread(target=rag_worker, args=(message_queue, thread_session_data, question_text, enable_multi_hop, search_budget, finished_signal))
        worker_thread.start()

        # SSE generator loop
        while not finished_signal.is_set() or not message_queue.empty():
            try:
                # Use a short timeout to remain responsive and check finished_signal
                msg = message_queue.get(timeout=0.1) 
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get('type') == 'finished':
                    break # Exit loop once finished signal is processed
            except queue.Empty:
                # Queue is empty, loop again to check finished_signal
                continue
            except Exception as e:
                 # Handle potential errors during SSE yield
                 app.logger.error(f"Error yielding SSE message: {e}")
                 break # Stop streaming on yield error

        # Ensure thread is joined if needed, though usually not necessary for SSE
        # worker_thread.join()

    return Response(stream_with_context(generate_response_stream()), mimetype='text/event-stream')

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    """处理语音文件上传和识别"""
    if "neo4j_config" not in session or not session.get("neo4j_connected"):
        return jsonify({"error": "Not connected to Neo4j. Please login."}), 403

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected."}), 400

    try:
        # 创建临时文件保存上传的音频
        temp_dir = tempfile.gettempdir()
        unique_filename = f"voice_{uuid.uuid4().hex}.wav"
        temp_audio_path = os.path.join(temp_dir, unique_filename)
        
        # 保存音频文件
        audio_file.save(temp_audio_path)
        
        # 进行语音识别
        app.logger.info(f"开始识别语音文件: {temp_audio_path}")
        recognized_text = recognize_voice(temp_audio_path)
        
        # 清理临时文件
        try:
            os.remove(temp_audio_path)
        except:
            pass  # 忽略清理错误
        
        if recognized_text:
            app.logger.info(f"语音识别成功: {recognized_text}")
            return jsonify({"success": True, "text": recognized_text})
        else:
            app.logger.warning("语音识别失败，未获得文本")
            return jsonify({"error": "语音识别失败，请重试或检查音频文件格式。"}), 400
    
    except Exception as e:
        app.logger.error(f"语音识别过程中出错: {e}", exc_info=True)
        # 确保清理临时文件
        try:
            if 'temp_audio_path' in locals():
                os.remove(temp_audio_path)
        except:
            pass
        return jsonify({"error": f"语音识别出错: {str(e)}"}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """处理图像文件上传和分割"""
    if "neo4j_config" not in session or not session.get("neo4j_connected"):
        return jsonify({"error": "Not connected to Neo4j. Please login."}), 403

    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image file selected."}), 400

    # 检查文件类型
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    if not (image_file.filename and '.' in image_file.filename and 
            image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "Unsupported file type. Please upload PNG, JPG, JPEG, GIF, BMP, or TIFF files."}), 400

    try:
        # 保存上传的图像
        app.logger.info(f"开始处理图像文件: {image_file.filename}")
        uploaded_path = image_segmentation_service.save_uploaded_image(image_file)
        
        if not uploaded_path:
            return jsonify({"error": "Failed to save uploaded image."}), 500
        
        # 进行图像分割
        app.logger.info(f"开始图像分割: {uploaded_path}")
        segmented_path, original_path, seg_info = image_segmentation_service.segment_image(
            uploaded_path,
            input_size=1024,
            iou_threshold=0.7,
            conf_threshold=0.25,
            better_quality=True,
            withContours=True,
            use_retina=True,
            mask_random_color=True
        )
        
        if not segmented_path:
            app.logger.error(f"图像分割失败: {seg_info}")
            return jsonify({"error": f"Image segmentation failed: {seg_info}"}), 500
        
        # 进行图像描述（仅分析分割后的图像）
        app.logger.info(f"开始图像描述: {segmented_path}")
        success, description = image_description_service.describe_medical_image(
            segmented_path
        )
        
        if not success:
            app.logger.warning(f"图像描述失败: {description}")
            description = "图像描述生成失败，但图像分割已完成。"
        
        # 返回成功结果
        app.logger.info(f"图像分割和描述完成: {segmented_path}")
        return jsonify({
            "success": True,
            "original_image": f"/uploads/{os.path.basename(original_path)}",
            "segmented_image": f"/segmented/{os.path.basename(segmented_path)}",
            "segmentation_info": seg_info,
            "description": description
        })
    
    except Exception as e:
        app.logger.error(f"图像分割过程中出错: {e}", exc_info=True)
        return jsonify({"error": f"Image segmentation error: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的图像文件"""
    return send_from_directory('static/uploads', filename)

@app.route('/segmented/<filename>')
def segmented_file(filename):
    """提供分割后的图像文件"""
    return send_from_directory('static/segmented', filename)

@app.route('/logout')
def logout():
    session.pop('neo4j_config', None)
    session['neo4j_connected'] = False
    return redirect(url_for('login'))

# --- Main ---
if __name__ == "__main__":
    # For development, Flask's reloader can cause issues with global state/threads
    # For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True, use_reloader=False)
    