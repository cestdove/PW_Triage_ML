# PW_Triage_ML

Project Work per il triage automatizzato di ticket di supporto aziendali con Machine Learning

Il sistema classifica ogni ticket lungo due assi dimensionali:

- `category`: `Administration`, `Sales`, `Technical`
- `priority`: `High`, `Medium`, `Low`

La pipeline usa del testo generato (`title` + `body`), vettorizzazione `TF-IDF` con unigrammi e bigrammi, un confronto di benchmark tra `LinearSVC` e `LogisticRegression` di cross-validation, e due modelli `LinearSVC` finali: uno per categoria e l'altro per priorità.

## Contenuto del Repository

- [1_generatore.py](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/1_generatore.py): genera un dataset sintetico di 3000 ticket con segnali contestuali, falsi allarmi, rumore sintattico e di etichetta 
- [2_pipeline_ml.py](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/2_pipeline_ml.py): compara due modelli lineari mediante cross-validation, addestra i modelli finali, e salva gli artefatti in `models/`
- [3_dashboard.py](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/3_dashboard.py): dashboard su Streamlit per batch di ticket, comparazione tra reale e predetto, e spiegabilità globale dei token
- [data/tickets.csv](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/data/tickets.csv): dataset CSV usato per l'addestramento e l'esecuzione della dashboard
- [models](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/models): modelli e vettorizzatore salvati
- [REPORT.md](/C:/Users/jnfpi/Desktop/Je/Mie/Università/PW_Triage_ML/REPORT.md): report descrittivo del progetto

## Stack delle librerie

- Python
- pandas
- numpy
- scikit-learn
- nltk
- joblib
- matplotlib
- seaborn
- streamlit

## Requisiti / dipendenze

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

Installare il corpus di NLTK richiesto per l'esecuzione:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Lo script di training usa la lista di stopword italiane di NLTK. Se il corpus è assente, `2_pipeline_ml.py` viene interrotto con un errore esplicito.

## Workflow consigliato

### 1. Generazione del dataset sintetico

```bash
python 1_generatore.py
```

Cosa fa:

- genera 3000 ticket sintetici
- crea o sovrascrive `data/tickets.csv`
- costruisce testo contenente rumore (typos, parole in maiuscolo, punteggiatura empatica)
- inietta template di criticità, segnali d'urgenza contestuali, e porzioni di falsi allarmi
- abbrevia alcune azioni per rendere i titoli più naturali e meno verbosi
- introduce rumore di etichettatura per simulare l'errore umano in task di dipartimento e priorità
- stampa a video le distribuzioni finali di categoria e priorità nella console

Formato del dataset:

- `id`
- `title`
- `body`
- `category`
- `priority`

### 2. Training dei modelli

```bash
python 2_pipeline_ml.py
```

Cosa fa:

- carica `data/tickets.csv`
- controlla la presenza delle colonne `title`, `body`, `category`, e `priority`
- crea `full_text` concatenando titolo e descrizione dei ticket
- applica `TfidfVectorizer` con:
  - `ngram_range=(1, 2)`
  - `min_df=5`
  - `max_df=0.85`
  - stopword italiane di NLTK
- esegue la cross-validation `StratifiedKFold` a 5-fold su category e priority
- compara `LinearSVC` e `LogisticRegression` con cross-validation
- stampa accuracy di fold, accuracy media, e deviazione standard per entrambi i modelli
- performa due train/test split stratificati con `test_size=0.2` e `random_state=42`
- addestra due modelli `LinearSVC(random_state=42, dual='auto')` 
- stampa accuracy e report di classificazione per category e priority
- mostra due matrici di confusione grafiche con `matplotlib` e `seaborn`
- salva gli artefatti dentro `models/`

Artefatti salvati:

- `models/svm_categ.joblib`
- `models/svm_prior.joblib`
- `models/tfidf_vectorizer.joblib`

Note:

- `LogisticRegression` è usato solo come benchmark durante cross-validation
- i modelli salvati e il modello utilizzato dalla dashboard restano i due classificatori `LinearSVC`

### 3. Esecuzione della Dashboard

```bash
streamlit run 3_dashboard.py
```

La dashboard carica autonomamente:

- `models/tfidf_vectorizer.joblib`
- `models/svm_categ.joblib`
- `models/svm_prior.joblib`
- `data/tickets.csv`

Se i modelli o il dataset risultano assenti, l'app mostra un errore ed interrompe il flusso di esecuzione.

## Feature della Dashboard

### 1. Export del Batch CSV 

La dashboard legge automaticamente `data/tickets.csv`, svolge le predizioni, e aggiunge:

- `predicted_category`
- `predicted_priority`

Il risultato è scaricabile come `triage_risultati_batch.csv`.

### 2. Confronto Reale VS Predetto

Se il file contiene anche le vere etichette `category` e `priority`, la dashboard mostra:

- comparazione tra distribuzioni di vero e predetto
- diagrammi a barre per category e priority
- matrici di confusione per category e priority

Questa sezione è utile per ispezionare il comportamento del modello sul batch corrente, non come una valutazione rigorosa separata dal training.

### 3. Spiegabilità globale

Per ogni classe di category e priority, la dashboard mostra:

- le 5 feature con il coefficiente positivo più alto nel modello lineare
- il peso numerico associato con ogni token
- una vista con la distribuzione dei token più influenti di ogni classe con i pesi corrispettivi

Questa spiegabilità:

- è globale, non specifica del singolo ticket
- proviene direttamente dai coefficienti di `LinearSVC`
- aiuta a comprendere quali token guidano maggiormente le predizioni
- rendono la separazione di classe più facile da interpretare attraverso comparazione visiva dei pesi dei token

## Struttura del progetto

```text
PW_Triage_ML/
|-- data/
|   `-- tickets.csv
|-- models/
|   |-- svm_categ.joblib
|   |-- svm_prior.joblib
|   `-- tfidf_vectorizer.joblib
|-- 1_generatore.py
|-- 2_pipeline_ml.py
|-- 3_dashboard.py
|-- README.md
|-- REPORT.md
`-- requirements.txt
```

## Note tecniche

- Il dataset è sintetico e costruito da vocabolari predefiniti, quindi le metriche riportate non rappresentano performance a livello di produzione.
- La classificazione di categoria è relativamente facile perché i domini di lessico sono abbastanza distinguibili. 
- Le predizioni di priorità sono più rumorose perché dipende dai segnali di contesto, template di criticità, ed errori di etichetta inseriti intenzionalmente.
- Il benchmark di `LogisticRegression` è usato come una comparazione rapida e non influenza il flusso finale.
- La dashboard non supporta ancora il caricamento manuale di CSV: utilizza soltanto il file locale `data/tickets.csv`.
- L'esecuzione della fase di training e la dashboard seguono lo stesso schema di preprocessing mediante il vettorizzatore salvato in comune.
- Matrici di confusione e report di classificazione sono basati sullo split di holdout, mentre la cross-validation offre una stima di accuracy più stabile.

## Limitazioni attuali

- Il benchmark è ancora limitato a un piccolo confronto tra `LinearSVC` e `LogisticRegression`
- Le metriche di performance sono stampate nella console ma non vengono ancora salvate in file di report strutturati
- La dashboard non offre ancora l'inferenza e l'analisi specifica di un singolo ticket
- La dashboard lavora solo sul file locale `data/tickets.csv` e non offre l'importazione manuale
- Il sistema è ancora organizzato come insieme di script lineari più che raccolta di moduli riutilizzabili
- Il preprocessing manca di lemmatizzazione

## Avvio rapido

Se vuoi utilizzare i modelli già inclusi nel repository:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
streamlit run 3_dashboard.py
```
