import os
import cv2
import numpy as np
import json
import argparse
import time
from datetime import datetime, timedelta
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import importlib.util
import torch
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector, ThresholdDetector

# Verifica se è disponibile PyTorch con supporto CUDA
def check_gpu_support():
    """
    Verifica se è disponibile il supporto GPU usando PyTorch.
    Restituisce: (disponibilità_cuda, nome_gpu, info_cuda)
    """
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_info = {
                'name': gpu_name,
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda
            }
            return True, gpu_name, cuda_info
    except Exception as e:
        print(f"Errore durante il controllo GPU con PyTorch: {str(e)}")
        return False, "GPU non disponibile", {}
    
    return False, "GPU non disponibile", {}

# Verifica supporto GPU
CUDA_AVAILABLE, GPU_NAME, CUDA_INFO = check_gpu_support()

# Configurazione predefinita
DEFAULT_CONFIG = {
    'max_frames': 2000,
    'sharpness_window': 7,
    'use_gpu': CUDA_AVAILABLE,
    'distribution_method': 'proportional',
    'min_scene_duration': 10.0,  # secondi
    'max_frames_per_scene': 5,
    'output_format': 'jpg',
    'jpg_quality': 95,
    'output_dir': 'frame_estratti',
    'scene_threshold': 27.0,
    'frames_per_10s': 1,  # Quanti frame estrarre ogni 10 secondi di scena
    'batch_size': 4 if CUDA_AVAILABLE else 1,  # Elabora più frame contemporaneamente
    'start_time': "00:00:00",  # Punto di inizio analisi (HH:MM:SS)
    'end_time': "",        # Punto di fine analisi (vuoto = fino alla fine del video)
    'use_time_range': False   # Se utilizzare l'intervallo temporale specificato
}

def format_time(seconds):
    """Converte secondi in formato HH:MM:SS"""
    td = timedelta(seconds=seconds)
    return str(td).split('.')[0]

def time_to_seconds(time_str):
    """Converte una stringa di tempo HH:MM:SS in secondi"""
    if not time_str:
        return 0
    
    parts = time_str.split(':')
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:  # MM:SS
        m, s = parts
        return int(m) * 60 + float(s)
    else:  # Solo secondi
        return float(time_str)

def seconds_to_time(seconds):
    """Converte secondi in formato HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"[:8]

# Function to install required packages if needed
def ensure_gpu_packages():
    """Verifica e suggerisce l'installazione dei pacchetti GPU se necessario"""
    if not CUDA_AVAILABLE:
        print("\n⚠️ Supporto GPU non rilevato!")
        print("Per abilitare l'accelerazione GPU:")
        print("1. Assicurati di avere installato i driver NVIDIA")
        print("2. Installa i pacchetti necessari con:")
        print("   pip install -r requirements_gpu.txt")
        print("\nNOTA: Lo script funzionerà comunque in modalità CPU")
        
        choice = input("\nVuoi continuare senza accelerazione GPU? (s/n): ").lower()
        if choice not in ['s', 'si', 'sì', 'y', 'yes']:
            print("\nInstallazione interrotta. Installa i requisiti GPU e riprova.")
            sys.exit(1)
    return CUDA_AVAILABLE

def calculate_sharpness_cpu(image):
    """
    Calcola il livello di nitidezza di un'immagine usando la varianza del Laplaciano (CPU).
    Valori più alti = immagine più nitida.
    """
    # Converti in scala di grigi se necessario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calcola il Laplaciano e la sua varianza
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def calculate_sharpness_pytorch(image, device=None):
    """
    Calcola il livello di nitidezza di un'immagine usando PyTorch.
    Valori più alti = immagine più nitida.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Converti in scala di grigi se necessario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Converti in tensore PyTorch e sposta su GPU
        img_tensor = torch.from_numpy(gray.astype(np.float32)).to(device)
        
        # Crea kernel Laplaciano
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # Aggiungi dimensioni batch e canale
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        # Applica il filtro Laplaciano
        with torch.no_grad():
            laplacian = torch.nn.functional.conv2d(img_tensor, laplacian_kernel, padding=1)
            
            # Calcola la varianza (indice di nitidezza)
            var = torch.var(laplacian).item()
            
        return var
    except Exception as e:
        print(f"⚠️ Errore GPU durante calcolo nitidezza: {str(e)}")
        print("⚠️ Cambio a modalità CPU...")
        # Fallback a calcolo CPU se ci sono errori
        return calculate_sharpness_cpu(image)

def find_sharpest_frame_cpu(cap, target_frame, window_size=5):
    """
    Cerca il frame più nitido in una finestra di frame intorno a quello target (CPU).
    Restituisce l'indice del frame più nitido e il frame stesso.
    """
    start_frame = max(0, target_frame - window_size)
    end_frame = target_frame + window_size
    
    best_sharpness = -1
    best_frame = None
    best_idx = -1
    
    for idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            sharpness = calculate_sharpness_cpu(frame)
            
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame
                best_idx = idx
    
    # Ritorna all'indice originale
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    return best_idx, best_frame, best_sharpness

def find_sharpest_frame_pytorch(cap, target_frame, window_size=5, use_gpu=False):
    """
    Cerca il frame più nitido in una finestra di frame intorno a quello target usando PyTorch.
    Restituisce l'indice del frame più nitido e il frame stesso.
    """
    # Se GPU non è richiesta o non disponibile, usa CPU
    if not use_gpu or not torch.cuda.is_available():
        return find_sharpest_frame_cpu(cap, target_frame, window_size)
    
    start_frame = max(0, target_frame - window_size)
    end_frame = target_frame + window_size
    
    best_sharpness = -1
    best_frame = None
    best_idx = -1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        for idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                sharpness = calculate_sharpness_pytorch(frame, device)
                
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame
                    best_idx = idx
        
        # Ritorna all'indice originale
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        return best_idx, best_frame, best_sharpness
    except Exception as e:
        print(f"⚠️ Errore durante l'elaborazione GPU: {str(e)}")
        print("⚠️ Passaggio a modalità CPU...")
        return find_sharpest_frame_cpu(cap, target_frame, window_size)

def process_frame_batch_pytorch(cap, frame_indices, window_size=5):
    """
    Processa un batch di frame in parallelo usando PyTorch.
    Restituisce una lista di tuple (frame_idx, best_idx, best_frame, sharpness)
    """
    if not torch.cuda.is_available():
        results = []
        for idx in frame_indices:
            best_idx, best_frame, sharpness = find_sharpest_frame_cpu(cap, idx, window_size)
            results.append((idx, best_idx, best_frame, sharpness))
        return results
    
    results = []
    device = torch.device("cuda")
    
    for frame_idx in frame_indices:
        # Cerca nella finestra del frame
        best_idx, best_frame, best_sharpness = find_sharpest_frame_pytorch(
            cap, frame_idx, window_size, True
        )
        results.append((frame_idx, best_idx, best_frame, best_sharpness))
    
    return results

def calculate_frames_for_scene(scene_duration, config):
    """
    Calcola il numero di frame da estrarre per una scena basandosi sulla durata.
    """
    if config['distribution_method'] == 'fixed':
        return min(config['max_frames_per_scene'], 1)
    else:  # proportional
        # Calcola quanti frame in base alla durata (1 frame ogni X secondi)
        frames_to_extract = max(1, int(scene_duration / (10.0 / config['frames_per_10s'])))
        return min(frames_to_extract, config['max_frames_per_scene'])

def extract_frames_from_scenes(video_path, config):
    """
    Estrae frame da scene rilevate in un video, scegliendo quelli più nitidi
    """
    # Ottieni nome del film dal percorso
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    print(f"\n{'='*70}")
    print(f"   ESTRAZIONE FRAME DA {video_name.upper()}")
    print(f"{'='*70}\n")
    
    # Crea la directory di output
    output_dir = os.path.expanduser(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Crea una sottocartella specifica per questo film
    film_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(film_output_dir, exist_ok=True)
    
    # Apri il video
    video = open_video(video_path)
    
    # Ottieni informazioni video  
    video_fps = video.frame_rate
    total_frames = video.duration.get_frames()
    video_duration = total_frames / video_fps
    
    # Converti i punti di inizio e fine in secondi e frame
    start_seconds = time_to_seconds(config['start_time']) if config['use_time_range'] else 0
    end_seconds = time_to_seconds(config['end_time']) if config['use_time_range'] and config['end_time'] else float('inf')
    
    # Converti in frame
    start_frame = int(start_seconds * video_fps) if config['use_time_range'] else 0
    end_frame = int(end_seconds * video_fps) if config['use_time_range'] and end_seconds < float('inf') else int(total_frames)
    
    # Se stiamo usando un intervallo di tempo, modifica il video passato al manager di scene
    if config['use_time_range']:
        # Crea frame timecodes per inizio e fine
        start_timecode = FrameTimecode(timecode=start_frame, fps=video_fps)
        end_timecode = FrameTimecode(timecode=min(end_frame, total_frames), fps=video_fps)
    else:
        start_timecode = None
        end_timecode = None
    
    print(f"📹 Informazioni video:")
    print(f"   • File: {video_filename}")
    print(f"   • FPS: {video_fps:.2f}")
    print(f"   • Durata: {format_time(video_duration)}")
    print(f"   • Frame totali: {total_frames}")
    
    # Mostra informazioni su intervalli temporali solo se abilitati
    if config['use_time_range']:
        if start_seconds > 0:
            print(f"   • Punto di inizio: {config['start_time']} ({format_time(start_seconds)})")
            print(f"   • Frame di inizio: {start_frame}")
        if end_seconds < float('inf'):
            print(f"   • Punto di fine: {config['end_time']} ({format_time(end_seconds)})")
            print(f"   • Frame di fine: {end_frame}")
        
        print(f"   • Porzione da analizzare: {format_time(end_seconds - start_seconds)}")
    
    # Mostra stato GPU
    if config['use_gpu'] and CUDA_AVAILABLE:
        print(f"   • Accelerazione GPU: ✅ Attiva")
        print(f"   • GPU rilevata: {GPU_NAME}")
        if 'total_memory' in CUDA_INFO:
            print(f"   • Memoria GPU: {CUDA_INFO['total_memory'] / (1024*1024):.1f} MB")
        print(f"   • Batch size: {config['batch_size']}")
    else:
        print(f"   • Accelerazione GPU: ❌ Disattiva")
    
    print(f"   • Finestra ricerca nitidezza: ±{config['sharpness_window']} frame")
    print(f"   • Metodo distribuzione: {config['distribution_method'].capitalize()}")
    print(f"   • Frame per 10s di scena: {config['frames_per_10s']}")
    print()
    
    # Inizializza detector per trovare scene
    scene_manager = SceneManager()
    detector = ContentDetector(threshold=config['scene_threshold'])
    scene_manager.add_detector(detector)
    
    print("🎬 Inizio rilevamento scene...")
    
    # SOLUZIONE: Se stiamo usando un intervallo temporale, dobbiamo prima assicurarci di settare i frame di inizio/fine
    # per analizzare solo quella porzione del video
    
    if config['use_time_range']:
        print(f"   • Analisi porzione video: {format_time(start_seconds)} - {format_time(min(end_seconds, video_duration))}")
        
        # Crea una sottosezione del video usando OpenCV per limitare l'area di analisi
        # Questa è una soluzione molto più efficiente e diretta
        
        # Apri il video con OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("⚠️ Errore nell'apertura del video.")
            return
        
        # Imposta la posizione iniziale del video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Crea un file video temporaneo per la porzione di interesse
        temp_path = os.path.join(os.path.dirname(video_path), f"temp_{int(time.time())}.mp4")
        
        # Ottiene i dettagli video necessari per il writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Crea un writer video per il file temporaneo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Numero di frame da estrarre
        frames_to_extract = end_frame - start_frame
        
        print(f"   • Preparazione file temporaneo per l'analisi della porzione...")
        
        # Leggi e scrivi i frame nella porzione di interesse
        frame_count = 0
        while cap.isOpened() and frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Mostra progresso
            if frame_count % 100 == 0:
                progress = (frame_count / frames_to_extract) * 100
                print(f"   • Preparazione: {progress:.1f}% completato", end="\r")
        
        # Rilascia le risorse
        cap.release()
        out.release()
        print(f"   • Preparazione completata: {frame_count} frame estratti")
        
        # Ora utilizza il file temporaneo per la rilevazione delle scene
        temp_video = open_video(temp_path)
        
        # Rileva le scene nel file temporaneo (che contiene solo la porzione di interesse)
        scene_manager.detect_scenes(temp_video, show_progress=True)
        
        # Ottieni le scene rilevate
        scene_list_relative = scene_manager.get_scene_list()
        
        # Converti le scene in coordinate assolute (rispetto al video originale)
        scene_list = []
        for scene in scene_list_relative:
            scene_start, scene_end = scene
            # Aggiungi l'offset di inizio
            absolute_start = FrameTimecode(timecode=scene_start.get_frames() + start_frame, fps=video_fps)
            absolute_end = FrameTimecode(timecode=scene_end.get_frames() + start_frame, fps=video_fps)
            scene_list.append((absolute_start, absolute_end))
        
        # Elimina il file temporaneo
        try:
            os.remove(temp_path)
            print(f"   • File temporaneo eliminato")
        except OSError as e:
            print(f"   • Errore nell'eliminazione del file temporaneo: {e}")
    else:
        print("   • Analisi intero video")
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
    
    # Ottieni lista scene rilevate
    num_scenes = len(scene_list)
    
    print(f"\n✅ Scene rilevate: {num_scenes}")
    print(f"🎯 Obiettivo massimo: {config['max_frames']} frame totali\n")
    
    # Calcola numero totale di frame da estrarre basandosi sulla durata delle scene
    total_estimated_frames = 0
    scene_frame_counts = []
    
    for scene in scene_list:
        scene_start_time, scene_end_time = scene
        scene_duration = (scene_end_time.get_frames() - scene_start_time.get_frames()) / video_fps
        frames_for_scene = calculate_frames_for_scene(scene_duration, config)
        scene_frame_counts.append(frames_for_scene)
        total_estimated_frames += frames_for_scene
    
    # Adatta il numero di frame se superiamo il limite
    if total_estimated_frames > config['max_frames']:
        reduction_factor = config['max_frames'] / total_estimated_frames
        scene_frame_counts = [max(1, int(count * reduction_factor)) for count in scene_frame_counts]
        total_estimated_frames = sum(scene_frame_counts)
    
    print(f"📊 Strategia di estrazione:")
    print(f"   • Frame previsti: {total_estimated_frames}")
    if config['distribution_method'] == 'proportional':
        print(f"   • Distribuzione: Proporzionale alla durata delle scene")
        print(f"   • Frame per 10s di scena: {config['frames_per_10s']}")
    else:
        print(f"   • Distribuzione: Fissa ({config['max_frames_per_scene']} per scena)")
    print()
    
    # Estrai frame
    frame_count = 0
    start_time = time.time()
    
    print(f"{'='*70}")
    print(f"   INIZIO ESTRAZIONE FRAME")
    print(f"{'='*70}\n")
    
    # Apri il video per estrazione
    cap = cv2.VideoCapture(video_path)
    
    # Prepara una coda di frame da processare per elaborazione batch
    for scene_idx, (scene, frames_to_extract) in enumerate(zip(scene_list, scene_frame_counts)):
        scene_start_time, scene_end_time = scene
        scene_start_frame = scene_start_time.get_frames()
        scene_end_frame = scene_end_time.get_frames()
        scene_duration = (scene_end_frame - scene_start_frame) / video_fps
        
        print(f"\n🎞️ Scena {scene_idx+1}/{num_scenes}")
        print(f"   • Inizio: {format_time(scene_start_time.get_seconds())}")
        print(f"   • Fine: {format_time(scene_end_time.get_seconds())}")
        print(f"   • Durata: {scene_duration:.2f}s")
        print(f"   • Frame da estrarre: {frames_to_extract}")
        
        # Se non ci sono frame da estrarre, passa alla prossima scena
        if frames_to_extract <= 0:
            continue
            
        # Calcola indici dei frame da estrarre
        if scene_end_frame - scene_start_frame <= frames_to_extract:
            # Prendi tutti i frame disponibili se ce ne sono meno del necessario
            frame_indices = list(range(int(scene_start_frame), int(scene_end_frame)))
        else:
            # Distribuisci uniformemente i frame nella scena
            step = (scene_end_frame - scene_start_frame) / frames_to_extract
            frame_indices = [int(scene_start_frame + i * step) for i in range(frames_to_extract)]
        
        # Se abbiamo raggiunto il limite massimo di frame, interrompi
        if frame_count >= config['max_frames']:
            print(f"\n🎯 Raggiunto limite massimo di {config['max_frames']} frame!")
            break
        
        # Estrazione batch con GPU
        if config['use_gpu'] and CUDA_AVAILABLE:
            # Suddividi i frame in batch
            batch_size = min(config['batch_size'], len(frame_indices))
            batches = [frame_indices[i:i + batch_size] for i in range(0, len(frame_indices), batch_size)]
            
            print(f"   • Elaborazione in {len(batches)} batch con GPU...")
            
            for batch_idx, batch in enumerate(batches):
                if frame_count >= config['max_frames']:
                    break
                    
                # Processa batch di frame
                results = process_frame_batch_pytorch(cap, batch, config['sharpness_window'])
                
                # Salva i risultati
                for i, (original_idx, best_idx, best_frame, sharpness) in enumerate(results):
                    if frame_count >= config['max_frames']:
                        break
                        
                    if best_frame is not None:
                        # Usa il tempo del frame realmente scelto
                        frame_time = best_idx / video_fps
                        filename = f"scena_{scene_idx+1:03d}_{frame_time:.2f}s.{config['output_format']}"
                        filepath = os.path.join(film_output_dir, filename)
                        
                        # Salva il frame
                        if config['output_format'].lower() == 'jpg':
                            cv2.imwrite(filepath, best_frame, [cv2.IMWRITE_JPEG_QUALITY, config['jpg_quality']])
                        else:
                            cv2.imwrite(filepath, best_frame)
                        
                        frame_count += 1
                        
                        # Calcola il delta rispetto al frame inizialmente scelto
                        frame_delta = best_idx - original_idx
                        delta_info = f"(delta: {frame_delta:+d})" if frame_delta != 0 else "(frame originale)"
                        
                        # Calcola ETA
                        if frame_count > 0:
                            elapsed_time = time.time() - start_time
                            frames_per_second = frame_count / elapsed_time
                            frames_left = min(config['max_frames'], total_estimated_frames) - frame_count
                            eta = frames_left / frames_per_second if frames_per_second > 0 else 0
                            
                            print(f"   ✓ Frame {i+1}/{len(batch)} del batch {batch_idx+1}: {filename} | Nitidezza: {sharpness:.2f} {delta_info} | ETA: {format_time(eta)}")
                        else:
                            print(f"   ✓ Frame {i+1}/{len(batch)} del batch {batch_idx+1}: {filename} | Nitidezza: {sharpness:.2f} {delta_info}")
                
                # Mostra progresso dopo ogni batch
                if frame_count > 0:
                    progress_percentage = frame_count / total_estimated_frames * 100
                    print(f"   📊 Progresso: {progress_percentage:.1f}% completato | {frame_count}/{total_estimated_frames} frame")
        else:
            # Estrazione standard (non parallela) 
            for i, frame_idx in enumerate(frame_indices):
                if frame_count >= config['max_frames']:
                    break
                    
                # Trova il frame più nitido nella finestra
                best_idx, best_frame, sharpness = find_sharpest_frame_cpu(
                    cap, frame_idx, config['sharpness_window']
                )
                
                if best_frame is not None:
                    # Usa il tempo del frame realmente scelto
                    frame_time = best_idx / video_fps
                    filename = f"scena_{scene_idx+1:03d}_{frame_time:.2f}s.{config['output_format']}"
                    filepath = os.path.join(film_output_dir, filename)
                    
                    # Salva il frame
                    if config['output_format'].lower() == 'jpg':
                        cv2.imwrite(filepath, best_frame, [cv2.IMWRITE_JPEG_QUALITY, config['jpg_quality']])
                    else:
                        cv2.imwrite(filepath, best_frame)
                    
                    frame_count += 1
                    
                    # Calcola il delta rispetto al frame inizialmente scelto
                    frame_delta = best_idx - frame_idx
                    delta_info = f"(delta: {frame_delta:+d})" if frame_delta != 0 else "(frame originale)"
                    
                    # Calcola tempo rimanente
                    elapsed_time = time.time() - start_time
                    if frame_count > 0:
                        avg_time_per_frame = elapsed_time / frame_count
                        frames_left = min(config['max_frames'], total_estimated_frames) - frame_count
                        eta = avg_time_per_frame * frames_left
                        print(f"   ✓ Frame {i+1}/{len(frame_indices)}: {filename} | Nitidezza: {sharpness:.2f} {delta_info} | ETA: {format_time(eta)}")
                    else:
                        print(f"   ✓ Frame {i+1}/{len(frame_indices)}: {filename} | Nitidezza: {sharpness:.2f} {delta_info}")
        
        # Aggiorna statistiche dopo ogni scena
        progress_percentage = (scene_idx + 1) / num_scenes * 100
        scenes_left = num_scenes - scene_idx - 1
        if scenes_left > 0 and frame_count > 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            estimated_total_time = total_estimated_frames / frames_per_second
            time_left = max(0, estimated_total_time - elapsed_time)
            
            print(f"   📊 Progresso: {progress_percentage:.1f}% completato | {frame_count}/{total_estimated_frames} frame | Tempo stimato: {format_time(time_left)}")
    
    # Chiudi il video
    cap.release()
    
    print(f"\n{'='*70}")
    print(f"   ESTRAZIONE COMPLETATA!")
    print(f"{'='*70}\n")
    print(f"✨ Frame totali estratti: {frame_count}")
    print(f"⏱️ Tempo totale: {format_time(time.time() - start_time)}")
    print(f"📁 Output directory: {film_output_dir}")
    print(f"\nI frame estratti sono pronti per addestrare il tuo modello LoRA!")

def find_video_files(directory):
    """Trova tutti i file video nella directory specificata"""
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files

def save_config(config, filepath):
    """Salva la configurazione in un file JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """Carica la configurazione da un file JSON"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            
            # Assicurati che tutte le chiavi del DEFAULT_CONFIG siano presenti
            for key in DEFAULT_CONFIG:
                if key not in loaded_config:
                    loaded_config[key] = DEFAULT_CONFIG[key]
                    
            return loaded_config
    return DEFAULT_CONFIG.copy()

def clear_screen():
    """Pulisce lo schermo del terminale"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Stampa l'intestazione del programma"""
    print(f"\n{'='*70}")
    print(f"   SCENE EXTRACTOR PER LORA TRAINING")
    if CUDA_AVAILABLE:
        print(f"   GPU: {GPU_NAME}")
    else:
        print(f"   GPU: Non disponibile")
    print(f"{'='*70}\n")

def print_menu(header, options, back_option=True):
    """Stampa un menu con opzioni numerate"""
    clear_screen()
    print_header()
    
    print(f"{header}\n")
    
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    
    if back_option:
        print(f"\n0. Torna indietro")
    
    return input("\nScelta: ")

def menu_intervallo_temporale(video_path, config):
    """Menu per impostare l'intervallo temporale di analisi"""
    video_filename = os.path.basename(video_path)
    
    # Ottieni durata del video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("\n⚠️ Impossibile aprire il video per verificare la durata.")
        return config
        
    # Ottieni informazioni video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    duration_str = format_time(duration)
    cap.release()
    
    clear_screen()
    print_header()
    print(f"INTERVALLO TEMPORALE - {video_filename}\n")
    print(f"Durata totale del video: {duration_str}\n")
    # Visualizza stato attuale
    if config['use_time_range']:
        start_time = config['start_time']
        end_time = config['end_time'] if config['end_time'] else "Fine del video"
        print(f"Intervallo attuale: {start_time} → {end_time}\n")
    else:
        print("Intervallo temporale: Non utilizzato (intero video)\n")
    
    # Opzioni menu
    options = [
        "Attiva/Disattiva intervallo temporale" + (" [Attivo]" if config['use_time_range'] else " [Disattivo]"),
        "Imposta punto di inizio",
        "Imposta punto di fine",
        "Usa intero video (reimposta intervallo)"
    ]
    
    choice = print_menu("Opzioni intervallo temporale:", options)
    
    if choice == '0':
        return config
    elif choice == '1':
        # Attiva/Disattiva intervallo
        config['use_time_range'] = not config['use_time_range']
        status = "attivato" if config['use_time_range'] else "disattivato"
        print(f"\n✅ Intervallo temporale {status}.")
        input("\nPremi INVIO per continuare...")
    elif choice == '2' and config['use_time_range']:
        # Imposta punto di inizio
        clear_screen()
        print_header()
        print(f"IMPOSTA PUNTO DI INIZIO - {video_filename}\n")
        print(f"Durata totale del video: {duration_str}")
        print(f"Formato: HH:MM:SS o MM:SS\n")
        
        new_value = input("Nuovo punto di inizio (lascia vuoto per inizio video): ")
        
        if new_value:
            try:
                seconds = time_to_seconds(new_value)
                # Verifica se l'orario è valido
                if seconds >= 0:
                    if seconds < duration:
                        config['start_time'] = new_value
                        print(f"\n✅ Punto di inizio impostato: {new_value} ({format_time(seconds)})")
                        
                        # Aggiorna anche il punto di fine se necessario
                        end_seconds = time_to_seconds(config['end_time']) if config['end_time'] else float('inf')
                        if end_seconds < seconds:
                            config['end_time'] = ""
                            print(f"⚠️ Punto di fine reimpostato alla fine del video (era prima del punto di inizio)")
                    else:
                        print(f"\n⚠️ Il punto di inizio è oltre la durata del video ({duration_str}).")
                else:
                    print("\n⚠️ Il tempo deve essere positivo.")
            except ValueError:
                print("\n⚠️ Formato tempo non valido. Usa HH:MM:SS o MM:SS.")
        else:
            config['start_time'] = "00:00:00"
            print("\n✅ Punto di inizio reimpostato all'inizio del video.")
        
        input("\nPremi INVIO per continuare...")
    elif choice == '3' and config['use_time_range']:
        # Imposta punto di fine
        clear_screen()
        print_header()
        print(f"IMPOSTA PUNTO DI FINE - {video_filename}\n")
        print(f"Durata totale del video: {duration_str}")
        print(f"Punto di inizio attuale: {config['start_time']}")
        print(f"Formato: HH:MM:SS o MM:SS\n")
        
        new_value = input("Nuovo punto di fine (lascia vuoto per fine video): ")
        
        if new_value:
            try:
                seconds = time_to_seconds(new_value)
                # Verifica se l'orario è valido
                if seconds >= 0:
                    # Verifica se il punto di fine è dopo il punto di inizio
                    start_seconds = time_to_seconds(config['start_time'])
                    if seconds > start_seconds:
                        config['end_time'] = new_value
                        print(f"\n✅ Punto di fine impostato: {new_value} ({format_time(seconds)})")
                        if seconds > duration:
                            print(f"⚠️ Nota: Il punto di fine è oltre la durata del video ({duration_str}), verrà usata la fine del video.")
                    else:
                        print(f"\n⚠️ Il punto di fine deve essere successivo al punto di inizio ({config['start_time']}).")
                else:
                    print("\n⚠️ Il tempo deve essere positivo.")
            except ValueError:
                print("\n⚠️ Formato tempo non valido. Usa HH:MM:SS o MM:SS.")
        else:
            config['end_time'] = ""
            print("\n✅ Punto di fine reimpostato alla fine del video.")
            
        input("\nPremi INVIO per continuare...")
    elif choice == '4':
        # Reimposta intervallo (usa intero video)
        config['use_time_range'] = False
        config['start_time'] = "00:00:00"
        config['end_time'] = ""
        print("\n✅ Intervallo temporale reimpostato (intero video).")
        input("\nPremi INVIO per continuare...")
    else:
        print("\n⚠️ Scelta non valida. Riprova.")
        input("\nPremi INVIO per continuare...")
    
    # Ritorna al menu intervallo
    return menu_intervallo_temporale(video_path, config)
    
def menu_principale():
    """Menu principale dell'applicazione"""
    # Verifica GPU all'avvio
    ensure_gpu_packages()
    
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    config = load_config(config_file)
    
    while True:
        clear_screen()
        print_header()
        
        # Trovare tutti i video
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_files = find_video_files(current_dir)
        
        if not video_files:
            print("❌ Nessun file video trovato nella directory corrente.")
            print("Per favore, aggiungi file video (mp4, mkv, avi, mov, wmv) nella stessa cartella dello script.")
            input("\nPremi INVIO per uscire...")
            sys.exit(1)
        
        print("Film trovati nella cartella:")
        for i, file in enumerate(video_files):
            print(f"{i+1}. {file}")
        
        choice = input("\nSeleziona un film (1-{}) o 'q' per uscire: ".format(len(video_files)))
        
        if choice.lower() == 'q':
            print("\nArrivederci!")
            sys.exit(0)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                selected_video = video_files[idx]
                video_path = os.path.join(current_dir, selected_video)
                
                # Menu per il video selezionato
                menu_video(video_path, config, config_file)
            else:
                print("\n⚠️ Scelta non valida. Riprova.")
                input("\nPremi INVIO per continuare...")
        except ValueError:
            print("\n⚠️ Inserisci un numero valido.")
            input("\nPremi INVIO per continuare...")

def menu_video(video_path, config, config_file):
    """Menu per il video selezionato"""
    video_filename = os.path.basename(video_path)
    
    while True:
        options = [
            f"Avvia estrazione con parametri predefiniti",
            f"Personalizza parametri",
            f"Imposta intervallo temporale",
            f"Visualizza descrizione dei parametri",
            f"Ripristina parametri predefiniti"
        ]
        
        choice = print_menu(f"Video selezionato: {video_filename}", options)
        
        if choice == '0':
            return
        elif choice == '1':
            # Avvia con parametri predefiniti
            extract_frames_from_scenes(video_path, config)
            input("\nPremi INVIO per tornare al menu...")
        elif choice == '2':
            # Personalizza parametri
            config = menu_personalizza_parametri(video_path, config, config_file)
        elif choice == '3':
            # Imposta intervallo temporale
            config = menu_intervallo_temporale(video_path, config)
            save_config(config, config_file)
        elif choice == '4':
            # Visualizza descrizione parametri
            menu_descrizione_parametri()
        elif choice == '5':
            # Ripristina parametri predefiniti
            config = DEFAULT_CONFIG.copy()
            save_config(config, config_file)
            print("\n✅ Parametri ripristinati ai valori predefiniti.")
            input("\nPremi INVIO per continuare...")
        else:
            print("\n⚠️ Scelta non valida. Riprova.")
            input("\nPremi INVIO per continuare...")

def menu_personalizza_parametri(video_path, config, config_file):
    """Menu per personalizzare i parametri"""
    while True:
        options = [
            f"Numero massimo di frame [{config['max_frames']}]",
            f"Finestra ricerca nitidezza [{config['sharpness_window']}]",
            f"Utilizzo GPU [{'Attivo' if config['use_gpu'] else 'Disattivo'}]",
            f"Distribuzione frame [{'Proporzionale' if config['distribution_method'] == 'proportional' else 'Fissa'}]",
            f"Frame ogni 10 secondi [{config['frames_per_10s']}]",
            f"Max frame per scena [{config['max_frames_per_scene']}]",
            f"Formato output [{config['output_format']}]",
            f"Qualità JPG [{config['jpg_quality']}]",
            f"Directory output [{config['output_dir']}]",
            f"Soglia rilevamento scene [{config['scene_threshold']}]",
            f"Dimensione batch GPU [{config['batch_size']}]",
            f"Avvia con questi parametri"
        ]
        
        choice = print_menu("PERSONALIZZA PARAMETRI", options)
        
        if choice == '0':
            # Salva configurazione e torna indietro
            save_config(config, config_file)
            return config
        elif choice == '1':
            # Numero massimo di frame
            try:
                new_value = int(input("\nInserisci nuovo valore per numero massimo di frame: "))
                if new_value > 0:
                    config['max_frames'] = new_value
                    print(f"\n✅ Valore aggiornato: {new_value}")
                else:
                    print("\n⚠️ Il valore deve essere maggiore di 0.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '2':
            # Finestra ricerca nitidezza
            try:
                new_value = int(input("\nInserisci nuovo valore per finestra ricerca nitidezza: "))
                if new_value > 0:
                    config['sharpness_window'] = new_value
                    print(f"\n✅ Valore aggiornato: {new_value}")
                else:
                    print("\n⚠️ Il valore deve essere maggiore di 0.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '3':
            # Utilizzo GPU
            if not CUDA_AVAILABLE:
                print("\n⚠️ GPU non disponibile sul tuo sistema.")
                input("\nPremi INVIO per continuare...")
                continue
                
            gpu_choice = input("\nAttivare accelerazione GPU? (s/n): ").lower()
            if gpu_choice in ['s', 'si', 'sì', 'y', 'yes']:
                config['use_gpu'] = True
                print("\n✅ Accelerazione GPU attivata.")
            elif gpu_choice in ['n', 'no']:
                config['use_gpu'] = False
                print("\n✅ Accelerazione GPU disattivata.")
            else:
                print("\n⚠️ Scelta non valida.")
        elif choice == '4':
            # Metodo distribuzione frame
            dist_choice = input("\nScegli metodo distribuzione (p=proporzionale, f=fisso): ").lower()
            if dist_choice in ['p', 'proporzionale']:
                config['distribution_method'] = 'proportional'
                print("\n✅ Distribuzione proporzionale impostata.")
            elif dist_choice in ['f', 'fisso']:
                config['distribution_method'] = 'fixed'
                print("\n✅ Distribuzione fissa impostata.")
            else:
                print("\n⚠️ Scelta non valida.")
        elif choice == '5':
            # Frame ogni 10 secondi
            try:
                new_value = float(input("\nInserisci numero di frame da estrarre ogni 10 secondi: "))
                if new_value > 0:
                    config['frames_per_10s'] = new_value
                    print(f"\n✅ Valore aggiornato: {new_value}")
                else:
                    print("\n⚠️ Il valore deve essere maggiore di 0.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '6':
            # Max frame per scena
            try:
                new_value = int(input("\nInserisci massimo numero di frame per scena: "))
                if new_value > 0:
                    config['max_frames_per_scene'] = new_value
                    print(f"\n✅ Valore aggiornato: {new_value}")
                else:
                    print("\n⚠️ Il valore deve essere maggiore di 0.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '7':
            # Formato output
            format_choice = input("\nScegli formato output (jpg/png): ").lower()
            if format_choice in ['jpg', 'jpeg']:
                config['output_format'] = 'jpg'
                print("\n✅ Formato JPG impostato.")
            elif format_choice in ['png']:
                config['output_format'] = 'png'
                print("\n✅ Formato PNG impostato.")
            else:
                print("\n⚠️ Formato non valido. Utilizza jpg o png.")
        elif choice == '8':
            # Qualità JPG
            try:
                new_value = int(input("\nInserisci qualità JPG (1-100): "))
                if 1 <= new_value <= 100:
                    config['jpg_quality'] = new_value
                    print(f"\n✅ Qualità impostata: {new_value}")
                else:
                    print("\n⚠️ La qualità deve essere tra 1 e 100.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '9':
            # Directory output
            new_value = input("\nInserisci directory di output: ")
            if new_value:
                config['output_dir'] = new_value
                print(f"\n✅ Directory impostata: {new_value}")
            else:
                print("\n⚠️ Directory non valida.")
        elif choice == '10':
            # Soglia rilevamento scene
            try:
                new_value = float(input("\nInserisci soglia rilevamento scene (1-100, valori più bassi=più scene): "))
                if 1 <= new_value <= 100:
                    config['scene_threshold'] = new_value
                    print(f"\n✅ Soglia impostata: {new_value}")
                else:
                    print("\n⚠️ La soglia deve essere tra 1 e 100.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '11':
            # Dimensione batch GPU
            if not CUDA_AVAILABLE:
                print("\n⚠️ GPU non disponibile. Impostazione ignorata.")
                input("\nPremi INVIO per continuare...")
                continue
                
            try:
                new_value = int(input("\nInserisci dimensione batch GPU (1-10): "))
                if 1 <= new_value <= 10:
                    config['batch_size'] = new_value
                    print(f"\n✅ Dimensione batch impostata: {new_value}")
                else:
                    print("\n⚠️ La dimensione deve essere tra 1 e 10.")
            except ValueError:
                print("\n⚠️ Inserisci un numero valido.")
        elif choice == '12':
            # Avvia con questi parametri
            save_config(config, config_file)
            extract_frames_from_scenes(video_path, config)
            input("\nPremi INVIO per tornare al menu...")
        else:
            print("\n⚠️ Scelta non valida. Riprova.")
            input("\nPremi INVIO per continuare...")
        
        # Salva configurazione
        save_config(config, config_file)
    
    return config

def menu_descrizione_parametri():
    """Mostra descrizione dettagliata dei parametri"""
    clear_screen()
    print_header()
    
    print("DESCRIZIONE DEI PARAMETRI\n")
    
    print("1. Numero massimo di frame")
    print("   Determina quanti frame estrarre in totale dal video.")
    print("   Valori più alti = dataset più grande, ma più tempo di elaborazione.")
    print("   Consigliato: 2000-5000 per la maggior parte dei film.")
    
    print("\n2. Finestra ricerca nitidezza")
    print("   Numero di frame prima e dopo da analizzare per trovare il frame più nitido.")
    print("   Valori più alti = frame più nitidi, ma più tempo di elaborazione.")
    print("   Consigliato: 5-10 per bilanciare qualità e velocità.")
    
    print("\n3. Utilizzo GPU")
    print("   Attiva/disattiva l'accelerazione GPU per le operazioni di elaborazione immagini.")
    print("   Consigliato: Attivo se disponibile una GPU NVIDIA compatibile.")
    
    print("\n4. Distribuzione frame")
    print("   'Proporzionale': estrae più frame per scene lunghe, meno per scene brevi.")
    print("   'Fissa': estrae lo stesso numero di frame per ogni scena.")
    print("   Consigliato: Proporzionale per una copertura migliore.")
    
    print("\n5. Frame ogni 10 secondi")
    print("   Quanti frame estrarre ogni 10 secondi di scena (in modalità proporzionale).")
    print("   Valori più alti = più frame per scene lunghe.")
    print("   Consigliato: 1-2 per la maggior parte dei casi.")
    
    print("\n6. Max frame per scena")
    print("   Limite superiore di frame estraibili da una singola scena.")
    print("   Consigliato: 5-10 per evitare troppe immagini simili.")
    
    print("\n7. Formato output")
    print("   'jpg': più compatto, leggermente inferiore in qualità.")
    print("   'png': qualità migliore, file più grandi.")
    print("   Consigliato: jpg per la maggior parte dei casi.")
    
    print("\n8. Qualità JPG")
    print("   Livello di qualità/compressione per i file JPG (1-100).")
    print("   Valori più alti = qualità migliore, file più grandi.")
    print("   Consigliato: 85-95 per un buon equilibrio.")
    
    print("\n9. Directory output")
    print("   La cartella dove salvare i frame estratti.")
    print("   Una sottocartella con il nome del film viene creata automaticamente.")
    
    print("\n10. Soglia rilevamento scene")
    print("   Sensibilità nel rilevare cambi di scena (1-100).")
    print("   Valori più bassi = più scene rilevate.")
    print("   Consigliato: 25-35 per film standard, 15-25 per film con molti tagli rapidi.")
    
    print("\n11. Dimensione batch GPU")
    print("   Numero di frame da elaborare contemporaneamente con la GPU.")
    print("   Valori più alti = maggiore velocità, ma più memoria GPU richiesta.")
    print("   Consigliato: 4-8 per GPU con 8GB+ di memoria, 2-4 per GPU con meno memoria.")
    
    print("\n12. Intervallo temporale")
    print("   Definisce una porzione specifica del video da analizzare.")
    print("   Utile per saltare titoli di testa/coda o concentrarsi su specifiche scene.")
    print("   Si imposta tramite il menu 'Imposta intervallo temporale'.")
    
    input("\nPremi INVIO per tornare al menu precedente...")

if __name__ == "__main__":
    try:
        menu_principale()
    except KeyboardInterrupt:
        print("\n\nEsecuzione interrotta dall'utente.")
    except Exception as e:
        print(f"\n\nErrore imprevisto: {str(e)}")
        input("\nPremi INVIO per uscire...")