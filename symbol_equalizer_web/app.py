from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import os
import uuid
import json
import threading
from datetime import datetime
from inference_engine import InferenceEngine
import pandas as pd
import numpy as np
from pathlib import Path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

PORT = 5001
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODELS_FOLDER'] = 'models'
CORS(app)

# Создаем папки
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    Path(folder).mkdir(exist_ok=True)

# Инициализируем движок инференса
inference_engine = InferenceEngine(models_dir=app.config['MODELS_FOLDER'])

# Загружаем конфигурацию моделей
MODELS_CONFIG = inference_engine.load_models_config()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html', models=MODELS_CONFIG)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Возвращает список доступных моделей"""
    return jsonify({
        'success': True,
        'models': MODELS_CONFIG
    })


import traceback  # Добавьте в импорты

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Загружает файл с символами"""
    try:
        print(f"\n[UPLOAD] Upload endpoint called")
        print(f"[UPLOAD] Request method: {request.method}")
        print(f"[UPLOAD] Request content type: {request.content_type}")
        print(f"[UPLOAD] Request files keys: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("[ERROR] No 'file' in request.files")
            return jsonify({
                'success': False, 
                'error': 'No file provided. Please select a CSV file.'
            }), 400
        
        file = request.files['file']
        print(f"[UPLOAD] File received: {file.filename}")
        print(f"[UPLOAD] File content type: {file.content_type}")
        print(f"[UPLOAD] File size: {file.content_length if hasattr(file, 'content_length') else 'unknown'}")
        
        if file.filename == '':
            print("[ERROR] Empty filename")
            return jsonify({
                'success': False, 
                'error': 'No file selected. Please choose a CSV file.'
            }), 400
        
        # Проверяем, что файл CSV
        if not file.filename.lower().endswith('.csv'):
            print(f"[ERROR] Invalid file type: {file.filename}")
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Only CSV files are allowed.'
            }), 400
        
        # Генерируем уникальный ID сессии
        session_id = str(uuid.uuid4())[:8]
        session['session_id'] = session_id
        
        # Обеспечиваем безопасное имя файла
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(file.filename)
        if safe_filename == '':
            safe_filename = f"data_{session_id}.csv"
        
        filename = f"{session_id}_{safe_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"[UPLOAD] Saving to: {filepath}")
        
        # Сохраняем файл
        file.save(filepath)
        
        # Проверяем, что файл сохранен
        if not os.path.exists(filepath):
            print(f"[ERROR] File not saved: {filepath}")
            return jsonify({
                'success': False, 
                'error': 'Failed to save file on server.'
            }), 500
        
        file_size = os.path.getsize(filepath)
        print(f"[UPLOAD] File saved successfully. Size: {file_size} bytes")
        
        # Читаем файл для предпросмотра
        try:
            # Читаем только первые 1000 строк для скорости
            chunks = []
            max_rows = 1000
            
            for chunk in pd.read_csv(filepath, header=None, chunksize=100, nrows=max_rows):
                chunks.append(chunk)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame()
            
            # Полный подсчет строк (это может быть медленно для больших файлов)
            total_rows = 0
            try:
                with open(filepath, 'r') as f:
                    total_rows = sum(1 for _ in f) - 1  # Минус заголовок если есть
            except:
                total_rows = len(df)
            
            stats = {
                'rows': total_rows,
                'columns': df.shape[1] if not df.empty else 0,
                'has_tx': df.shape[1] >= 4 if not df.empty else False,
                'preview': df.head(10).values.tolist() if not df.empty else []
            }
            
            print(f"[UPLOAD] File stats: {stats['rows']} rows, {stats['columns']} cols")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'filename': filename,
                'original_filename': file.filename,
                'file_size': file_size,
                'file_path': filepath,
                'stats': stats,
                'message': f'File uploaded successfully. {stats["rows"]:,} symbols loaded.'
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process CSV: {str(e)}")
            print(traceback.format_exc())
            
            # Удаляем поврежденный файл
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': False, 
                'error': f'Failed to process CSV file: {str(e)}. Please ensure it is a valid CSV file.'
            }), 400
            
    except Exception as e:
        print(f"[CRITICAL ERROR] Upload failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500
    
    
@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Запускает инференс"""
    data = request.json
    session_id = data.get('session_id')
    model_id = data.get('model_id')
    filename = data.get('filename')
    
    if not all([session_id, model_id, filename]):
        return jsonify({'success': False, 'error': 'Missing parameters'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_id = f"{session_id}_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Запускаем инференс в отдельном потоке
    def inference_task():
        try:
            # Выполняем инференс
            result = inference_engine.run_inference(
                model_id=model_id,
                input_file=filepath,
                session_id=session_id,
                result_id=result_id
            )
            
            # Сохраняем результат
            result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error in inference: {e}")
    
    thread = threading.Thread(target=inference_task)
    thread.start()
    
    return jsonify({
        'success': True,
        'result_id': result_id,
        'message': 'Inference started in background'
    })

@app.route('/api/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Получает результаты инференса"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(result_file):
        return jsonify({'success': False, 'error': 'Result not found'})
    
    try:
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Генерируем визуализации
        if result.get('success'):
            # Constellation plot
            constellation_plot = create_constellation_plot(result)
            # BER comparison
            ber_plot = create_ber_plot(result)
            # Time series
            time_plot = create_time_series_plot(result)
            
            result['plots'] = {
                'constellation': constellation_plot,
                'ber_comparison': ber_plot,
                'time_series': time_plot
            }
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download/<result_id>', methods=['GET'])
def download_results(result_id):
    """Скачивает результаты в CSV"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(result_file):
        return jsonify({'success': False, 'error': 'Result not found'})
    
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    # Создаем CSV
    if result.get('success'):
        df_pred = pd.DataFrame(result['predictions']['symbols'], columns=['I_pred', 'Q_pred'])
        df_bits = pd.DataFrame(result['predictions']['bits'], columns=['b0', 'b1', 'b2', 'b3'])
        df = pd.concat([df_pred, df_bits], axis=1)
        
        # Сохраняем во временный файл
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'equalized_results_{result_id}.csv'
        )
    
    return jsonify({'success': False, 'error': 'No predictions available'})

def create_constellation_plot(result):
    """Создает график созвездия"""
    rx_symbols = result['data']['rx_symbols']
    pred_symbols = result['predictions']['symbols']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Received Symbols', 'Equalized Symbols'),
        horizontal_spacing=0.15
    )
    
    # Received symbols
    fig.add_trace(
        go.Scatter(
            x=rx_symbols['I'][:1000],  # Ограничиваем для производительности
            y=rx_symbols['Q'][:1000],
            mode='markers',
            marker=dict(size=3, opacity=0.5, color='blue'),
            name='RX'
        ),
        row=1, col=1
    )
    
    # Equalized symbols
    fig.add_trace(
        go.Scatter(
            x=[s[0] for s in pred_symbols[:1000]],
            y=[s[1] for s in pred_symbols[:1000]],
            mode='markers',
            marker=dict(size=3, opacity=0.5, color='green'),
            name='Equalized'
        ),
        row=1, col=2
    )
    
    # Конфигурация
    fig.update_layout(
        height=500,
        showlegend=True,
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text="I", row=1, col=1)
    fig.update_yaxes(title_text="Q", row=1, col=1)
    fig.update_xaxes(title_text="I", row=1, col=2)
    fig.update_yaxes(title_text="Q", row=1, col=2)
    
    return json.loads(fig.to_json())

def create_ber_plot(result):
    """Создает график сравнения BER"""
    if 'metrics' not in result or 'baseline_ber' not in result['metrics']:
        return None
    
    metrics = result['metrics']
    
    fig = go.Figure(data=[
        go.Bar(
            name='BER',
            x=['Baseline', 'Equalized'],
            y=[metrics['baseline_ber'], metrics['equalized_ber']],
            text=[f"{metrics['baseline_ber']:.2e}", f"{metrics['equalized_ber']:.2e}"],
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4']
        )
    ])
    
    fig.update_layout(
        title='BER Comparison',
        yaxis_type="log",
        yaxis_title="BER (log scale)",
        height=400,
        template='plotly_dark'
    )
    
    return json.loads(fig.to_json())

def create_time_series_plot(result):
    """Создает график временного ряда"""
    if 'data' not in result or 'rx_symbols' not in result['data']:
        return None
    
    rx_i = result['data']['rx_symbols']['I'][:500]
    pred_i = [s[0] for s in result['predictions']['symbols'][:500]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=rx_i,
        mode='lines',
        name='Received I',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        y=pred_i,
        mode='lines',
        name='Equalized I',
        line=dict(color='green', width=1)
    ))
    
    fig.update_layout(
        title='Time Series (I component)',
        xaxis_title='Symbol Index',
        yaxis_title='Amplitude',
        height=400,
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return json.loads(fig.to_json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)