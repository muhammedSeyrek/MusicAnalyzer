from flask import Flask, render_template, request, jsonify, Response, copy_current_request_context
from werkzeug.utils import secure_filename
import os
import json
import time
import threading
import queue
from analyzer import analyze_music

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global progress tracking
progress_queues = {}

def progress_callback(queue_id):
    def callback(progress):
        progress_queues[queue_id].put(progress)
    return callback

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/progress')
def progress():
    """Server-sent events handler for progress updates"""
    queue_id = request.args.get('id', '')
    
    # Şu anki request context'i kopyalayalım
    @copy_current_request_context
    def generate():
        if queue_id in progress_queues:
            q = progress_queues[queue_id]
            try:
                while True:
                    progress = q.get(timeout=60)  # 60 saniye timeout
                    if progress == -1:  # Signal for completion
                        yield f"data: 100\n\n"
                        break
                    yield f"data: {progress}\n\n"
            except queue.Empty:
                # Timeout olursa bir -1 gönderip bağlantıyı kapatalım
                yield f"data: -1\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# JSON serileştirme için özel encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, int, float, str)):
            return obj
        # NumPy tiplerine özel dönüşüm
        if hasattr(obj, 'item'):
            return obj.item()
        # Diğer iterable nesneler için liste dönüşümü yap
        try:
            return list(obj)
        except:
            return str(obj)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    # Generate a unique ID for this analysis job
    queue_id = str(time.time())
    progress_queues[queue_id] = queue.Queue()
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Run analysis in a separate thread to not block the response
    def run_analysis():
        try:
            # Call the analysis function with progress callback
            result = analyze_music(filepath, progress_callback(queue_id))
            
            # Signal completion and cleanup
            progress_queues[queue_id].put(-1)
            if queue_id in progress_queues:
                del progress_queues[queue_id]
                
            # Save results to a JSON file that can be retrieved later
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_result.json")
            with open(result_path, 'w') as f:
                # Özel JSON encoder kullanarak tüm veri tiplerini işle
                json.dump(result, f, cls=CustomJSONEncoder)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            # Hata durumunda kuyruğu temizle
            if queue_id in progress_queues:
                progress_queues[queue_id].put(-1)
    
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True  # Ana program kapandığında thread de kapansın
    thread.start()
    
    return jsonify({"status": "processing", "queue_id": queue_id, "filename": filename})

@app.route('/result/<filename>')
def get_result(filename):
    """Retrieve analysis results for a file"""
    base_filename = os.path.splitext(filename)[0]
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_filename}_result.json")
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            try:
                return jsonify(json.load(f))
            except json.JSONDecodeError as e:
                return jsonify({"error": f"JSON decode error: {str(e)}"})
    else:
        return jsonify({"error": "Results not found"})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    # Benzersiz bir dosya adı oluştur (zaman damgası ekleyerek)
    timestamp = int(time.time())
    base_filename = secure_filename(file.filename)
    filename_parts = os.path.splitext(base_filename)
    unique_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Dosya analizi için arka plan thread'i oluştur
    def run_analysis():
        try:
            # İlerleme takibi olmadan analiz et (SSE kullanmıyoruz artık)
            result = analyze_music(filepath, None)
            
            # Sonuçları bir JSON dosyasına kaydet
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(unique_filename)[0]}_result.json")
            with open(result_path, 'w') as f:
                # Özel JSON encoder kullanarak tüm veri tiplerini işle
                json.dump(result, f, cls=CustomJSONEncoder)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            # Hata durumunda boş bir sonuç dosyası oluştur
            error_result = {"error": f"Analysis failed: {str(e)}"}
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(unique_filename)[0]}_result.json")
            with open(result_path, 'w') as f:
                json.dump(error_result, f)
    
    # Thread'i başlat ve hemen yanıt döndür
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    # Sadece dosya adını döndür, sonuçlar hazır olduğunda istemci kontrol edecek
    return jsonify({"status": "processing", "filename": unique_filename})







if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)