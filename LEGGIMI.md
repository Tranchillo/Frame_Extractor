# Estrattore Immagini per Addestramento LoRA

Questo programma consente di estrarre automaticamente fotogrammi di alta qualità da file video per creare dataset di addestramento per modelli LoRA. Il programma divide automaticamente il video in scene, seleziona i fotogrammi più nitidi e li salva in formato immagine.

## Caratteristiche Principali

- **Rilevamento automatico delle scene**: Identifica i cambi di scena nel video
- **Selezione intelligente dei fotogrammi**: Sceglie i frame più nitidi per ogni scena
- **Accelerazione GPU**: Supporto per CUDA per elaborazione più veloce (opzionale)
- **Interfaccia intuitiva**: Menu interattivo a riga di comando
- **Configurazione personalizzabile**: Regola tutti i parametri per adattarli alle tue esigenze

## Requisiti di Sistema

- Python 3.7 o superiore
- OpenCV
- PyTorch (per accelerazione GPU)
- SceneDetect
- Altri pacchetti Python (elencati in requirements.txt)
- NVIDIA GPU con driver compatibili (opzionale, per accelerazione GPU)

## Installazione

### 1. Clona o scarica questo repository:

```bash
git clone https://github.com/Tranchillo/Frame_Extractor.git
cd Frame_Extractor
```

### 2. Creazione di un ambiente virtuale (consigliato)

È consigliabile utilizzare un ambiente virtuale Python per evitare conflitti con altre installazioni:

```bash
# Creazione dell'ambiente virtuale
python -m venv venv

# Attivazione dell'ambiente virtuale
# Su Windows:
venv\Scripts\activate
# Su macOS/Linux:
source venv/bin/activate
```

Una volta attivato l'ambiente virtuale, dovresti vedere `(venv)` all'inizio della riga di comando, indicando che stai lavorando nell'ambiente isolato.

### 3. Installazione delle dipendenze

#### Verifica se hai una GPU NVIDIA disponibile

Esegui il comando `nvidia-smi` per verificare se hai una GPU NVIDIA e quali driver sono installati:

```bash
nvidia-smi
```

Se il comando funziona, dovresti vedere un output simile a questo:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.146.02    Driver Version: 535.146.02    CUDA Version: 12.2   |
| ...                                                                          |
```

Prendi nota della versione CUDA riportata (nell'esempio è 12.2).

#### Installazione PyTorch con supporto CUDA

In base alla versione CUDA mostrata da `nvidia-smi`, scegli il comando di installazione corretto:

- Per CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Per CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Installazione di altre dipendenze

Installa le altre dipendenze necessarie:

```bash
pip install opencv-python numpy tqdm scenedetect
```

## Utilizzo

### Avvio del programma

Posiziona i file video (.mp4, .mkv, .avi, ecc.) nella stessa cartella del programma, quindi esegui:

```bash
# Assicurati che l'ambiente virtuale sia attivato
# Su Windows:
venv\Scripts\activate
# Su macOS/Linux:
source venv/bin/activate

# Avvia il programma
python Estrattore_Immagini.py
```

### Menu principale

All'avvio, il programma visualizzerà un menu con i video disponibili nella cartella corrente. Seleziona un numero per scegliere il video da elaborare.

### Menu del video selezionato

Dopo aver selezionato un video, apparirà un menu con le seguenti opzioni:

1. **Avvia estrazione con parametri predefiniti**: Inizia subito l'estrazione dei frame
2. **Personalizza parametri**: Modifica i parametri di estrazione
3. **Imposta intervallo temporale**: Scegli una porzione specifica del video
4. **Visualizza descrizione dei parametri**: Informazioni dettagliate sui parametri
5. **Ripristina parametri predefiniti**: Reimposta tutte le impostazioni

### Parametri personalizzabili

Tutti i parametri possono essere regolati in base alle tue esigenze:

- **Numero massimo di frame**: Quanti frame estrarre in totale
- **Finestra ricerca nitidezza**: Per selezionare i frame più nitidi
- **Utilizzo GPU**: Attiva/disattiva l'accelerazione hardware
- **Distribuzione frame**: Proporzionale o fissa per ogni scena
- **Frame ogni 10 secondi**: Densità di campionamento per scene lunghe
- **Max frame per scena**: Limite per evitare troppe immagini simili
- **Formato output**: JPG o PNG
- **Qualità JPG**: Livello di compressione per i file JPG
- **Directory output**: Dove salvare i frame estratti
- **Soglia rilevamento scene**: Sensibilità nel rilevare i cambi di scena
- **Dimensione batch GPU**: Per ottimizzare l'elaborazione parallela

### Intervallo temporale

Puoi anche impostare un intervallo temporale specifico per concentrarti su una parte particolare del video:

1. **Attiva/disattiva intervallo temporale**: Abilita l'uso di un intervallo
2. **Imposta punto di inizio**: In formato HH:MM:SS
3. **Imposta punto di fine**: In formato HH:MM:SS
4. **Usa intero video**: Reimposta per utilizzare tutto il video

## Output

I frame estratti vengono salvati nella directory specificata (predefinita: `frame_estratti`) in una sottocartella con il nome del file video. Ogni frame è nominato con il numero della scena e il timestamp.

## Suggerimenti per ottenere risultati migliori

1. **Impostazioni per video di alta qualità**:
   - Aumenta il numero massimo di frame a 3000-5000
   - Usa la distribuzione proporzionale
   - Imposta una finestra di ricerca nitidezza più ampia (7-10)

2. **Impostazioni per prestazioni veloci**:
   - Riduci il numero massimo di frame a 1000-2000
   - Attiva l'accelerazione GPU se disponibile
   - Usa una finestra di ricerca nitidezza più piccola (3-5)

3. **Estrazione di scene specifiche**:
   - Usa l'opzione "Imposta intervallo temporale"
   - Specifica i punti esatti di inizio e fine in formato HH:MM:SS

## Risoluzione dei problemi

### Problemi GPU

Se riscontri problemi con l'accelerazione GPU:

1. **Verifica la compatibilità**: Assicurati di aver installato PyTorch con la versione CUDA corretta per i tuoi driver
2. **Disattiva l'accelerazione GPU**: Se i problemi persistono, puoi sempre usare la modalità CPU
3. **Aggiorna i driver**: A volte potrebbe essere necessario aggiornare i driver NVIDIA

### Errori di memoria

Se il programma va in errore per problemi di memoria:

1. **Riduci la dimensione batch GPU**: Prova con valori più bassi
2. **Elabora meno frame**: Diminuisci il numero massimo di frame
3. **Processa un intervallo più piccolo**: Usa l'opzione intervallo temporale per elaborare il video in parti

## Versione inglese

È disponibile anche una versione in inglese del programma: `Frame_Extractor.py`. Funziona esattamente allo stesso modo ma con tutti i menu e i messaggi in inglese.

## Licenza

Questo software è distribuito con licenza MIT.

---

Per domande, suggerimenti o segnalazioni di bug, apri un issue su GitHub: https://github.com/Tranchillo/Frame_Extractor/issues