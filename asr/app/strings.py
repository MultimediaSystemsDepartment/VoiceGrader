"""strings.py
A dictionary of blingual scripts for the application.
"""

# Table of strings, arranged by strings.

__STRING_TABLE = {
    'service_name': {
        'pl': 'Usługa automatycznej transkrypcji mowy na tekst',
        'en': 'Speech-to-text service',
    },
    'affiliation': {
        'pl': 'Katedra Systemów Multimedialnych, Politechnika Gdańska, Wydział ETI',
        'en': 'Department of Multimedia Systems, Gdańsk University of Technology',
    },
    'admed_caise': {
        'pl': 'Usługa opracowana w ramach projektu badawczego ADMEDVOICE, uruchomiona w ramach projektu CAISE.',
        'en': 'Service developed within the ADMEDVOICE scientific project, implemented within the CAISE project.',
    },
    'fileproc': {
        'pl': 'Przetwarzanie plików',
        'en': 'Audio file processing',
    },
    'liveproc': {
        'pl': 'Przetwarzanie na żywo z mikrofonu',
        'en': 'Live microphone stream processing',
    },
    'device_select': {
        'pl': 'Wejściowe urządzenie dźwiękowe',
        'en': 'Input audio device',
    },
    'lang_select': {
        'pl': 'Język rozpoznawania mowy',
        'en': 'Recognition language',
    },
    'lang_auto': {
        'pl': 'Automatyczny wybór',
        'en': 'Automatic detection',
    },
    'input_data': {
        'pl': 'Dane wejściowe',
        'en': 'Input data',
    },
    'input_file': {
        'pl': 'Plik dźwiękowy (WAV, MP3)',
        'en': 'Audio file (WAV, MP3)',
    },
    'mic_record': {
        'pl': 'Nagrywanie z mikrofonu',
        'en': 'Recording from microphone',
    },
    'record': {
        'pl': 'Nagrywanie',
        'en': 'Record',
    },
    'clear_btn': {
        'pl': 'Wyczyść',
        'en': 'Clear',
    },
    'preview': {
        'pl': 'Podgląd pliku dźwiękowego',
        'en': 'Audio preview',
    },
    'ref_text': {
        'pl': 'Referencyjny tekst',
        'en': 'Reference text',
    },
    'ref_placeholder': {
        'pl': 'Tekst do porównania z transkryptem (opcjonalny)',
        'en': 'Reference text to compare (optional)',
    },
    'start_processing': {
        'pl': 'Uruchom przetwarzanie',
        'en': 'Start processing',
    },
    'status_wait': {
        'pl': 'Oczekiwanie na wczytanie pliku dźwiękowego lub zarejestrowanie dźwięku z mikrofonu.',
        'en': 'Waiting for opening audio file or recording audio from microphone.',
    },
    'transcript_header': {
        'pl': 'Transkrypt mowy',
        'en': 'Recognized text',
    },
    'clipboard_copy': {
        'pl': 'Skopiuj tekst do schowka',
        'en': 'Copy text to clipboard',
    },
    'save_text': {
        'pl': 'Zapisz tekst do pliku',
        'en': 'Save text to file',
    },
    'load_text': {
        'pl': 'Wczytaj tekst z pliku',
        'en': 'Load text from file',
    },
    'tts_send': {
        'pl': 'Prześlij rozpoznany tekst do usługi syntezy mowy',
        'en': 'Send the recognized text to text-to-speech service',
    },
    'metrics': {
        'pl': 'Metryki przetwarzania nagrania',
        'en': 'File processing metrics',
    },
    'logprob': {
        'pl': 'Stopień pewności transkrypcji (0,0-1,0; większy: lepiej)',
        'en': 'Transcription certainty (0.0-1.0; higher = better)',
    },
    'wer': {
        'pl': 'WER (Word Error Rate; mniejszy: lepiej)',
        'en': 'WER (Word Error Rate; lower = better)',
    },
    'cer': {
        'pl': 'CER (Character Error Rate; mniejszy: lepiej)',
        'en': 'CER (Character Error Rate; lower = better)',
    },
    'mos': {
        'pl': 'Stopień jakości nagrania (skala MOS, 1,0-5,0; większa: lepiej)',
        'en': 'Recording quality (MOS scale, 1.0-5.0; higher = better)',
    },
    'snr': {
        'pl': 'Poziom SNR (mowa-szum; większy: lepiej)',
        'en': 'SNR (speech-to-noise ratio; higher = better)',
    },
    'gpu_mean': {
        'pl': 'Średnie wykorzystanie GPU',
        'en': 'Mean GPU utilization',
    },
    'gpu_max': {
        'pl': 'Maksymalne wykorzystanie GPU',
        'en': 'Max GPU utilization',
    },
    'vram_mean': {
        'pl': 'Średnie wykorzystanie pamięci VRAM',
        'en': 'Mean VRAM utilization',
    },
    'vram_max': {
        'pl': 'Maksymalne wykorzystanie pamięci VRAM',
        'en': 'Max VRAM utilization',
    },
    'livetrans': {
        'pl': 'Transkrypcja na żywo sygnału z mikrofonu',
        'en': 'Live transcription of microphone audio',
    },
    'ready': {
        'pl': 'Gotowe',
        'en': 'Ready',
    },
    'stats1': {
        'pl': 'Statystyki obciążenia urządzenia',
        'en': 'Statistics of device utilization',
    },
    'stats2': {
        'pl': 'odświeżane co 5 sekund',
        'en': 'refreshed every 5 seconds',
    },
    'ksmpg': {
        'pl': 'Katedra Systemów Multimedialnych, Politechnika Gdańska, Wydział Elektroniki, Telekomunikacji i Informatyki.',
        'en': 'Department of Multimedia Systems, Gdańsk University of Technology, Faculty of Electronics, Telecommunication and Informatics.',
    },
    'projects': {
        'pl': 'Usługa zrealizowana w ramach poniższych projektów.',
        'en': 'Service created within the following projects.',
    },
    'caise': {
        'pl': 'Chmurowa Platforma do wytwarzania uniwersalnych usług inteligentnych dla różnych obszarów zastosowań. '
        'Instytucja wdrażająca: Narodowe Centrum Badań i Rozwoju (NCBR). '
        'Program finansujący: KPO IPCEI CIS C 3.1.1 Important Project of Common European Interest. '
        'Nr projektu: KPOD.05.10-IW.10-0005/24.',
        'en': 'Cloud platform for creating universal intelligent services for various application areas. '
        'Implementing Institution: National Centre for Research and Development (NCBR). '
        'Financing Program: KPO IPCEI CIS C 3.1.1 Important Project of Common European Interest. '
        'Project No.: KPOD.05.10-IW.10-0005/24.',
    },
    'admedvoice': {
        'pl': 'Adaptacyjny system inteligentnego przetwarzania mowy lekarzy wraz ze '
        'strukturalizacją wyników badań i wspomaganiem procesu terapeutycznego. '
        'Projekt współfinansowany przez Narodowe Centrum Badań i Rozwoju w ramach programu '
        'Infostrateg IV, nr projektu: INFOSTRATEG-IV/0003/2022.',
        'en': 'Adaptive intelligent speech processing system of medical personnel with the '
        'structuring of test results and support of therapeutic process. '
        'The project was co-financed by the National Centre for Research and Development '
        'under the Infostrateg IV program, project number: INFOSTRATEG-IV/0003/2022.',
    },
    'device_ready': {
        'pl': 'Urządzenie dźwiękowe jest gotowe',
        'en': 'Audio device is ready',
    },
    'device_no_access': {
        'pl': 'Brak dostępu do urządzeń dźwiękowych',
        'en': 'No access to audio devices',
    },
    'no_audio_devices': {
        'pl': 'Brak dostępnych wejściowych urządzeń dźwiękowych.',
        'en': 'No input audio device available.',
    },
    'in_recording': {
        'pl': 'Nagrywanie...',
        'en': 'Recording...',
    },
    'rec_done': {
        'pl': 'Nagranie ukończone',
        'en': 'Recording finished',
    },
    'rec_done_status': {
        'pl': 'Zakończono nagrywanie pliku. Gotowy do przetwarzania.',
        'en': 'Audio recording finished. Ready for processing.',
    },
    'proc_prepare': {
        'pl': 'Przesyłanie nagrania do aplikacji.',
        'en': 'Uploading audio file.',
    },
    'proc_started': {
        'pl': 'Rozpoczęto przetwarzanie.',
        'en': 'Processing started.',
    },
    'proc_finished': {
        'pl': 'Zakończono przetwarzanie w czasie',
        'en': 'Processing finished in',
    },
    'proc_error': {
        'pl': 'Błąd przetwarzania.',
        'en': 'Processing error.',
    },
    'excellent': {
        'pl': 'ZNAKOMITY',
        'en': 'EXCELLENT',
    },
    'good': {
        'pl': 'DOBRY',
        'en': 'GOOD',
    },
    'fair': {
        'pl': 'DOSTATECZNY',
        'en': 'FAIR',
    },
    'poor': {
        'pl': 'SŁABY',
        'en': 'POOR',
    },
    'bad': {
        'pl': 'ZŁY',
        'en': 'BAD',
    },
    'unacceptable': {
        'pl': 'NIEAKCEPTOWALNY',
        'en': 'UNACCEPTABLE',
    },
    'server_wait': {
        'pl': 'Oczekiwanie na połączenie z serwerem...',
        'en': 'Waiting for the server...',
    },
    'server_finished': {
        'pl': 'Przetwarzanie zakończone.',
        'en': 'Processing finished.',
    },
    'server_busy': {
        'pl': 'Serwer transkrypcji zajęty, prosimy spróbować później.',
        'en': 'Server busy, please try later.',
    },
    'server_error': {
        'pl': 'Błąd serwera.',
        'en': 'Server error.',
    },
    'server_ready': {
        'pl': 'Rozpoznawanie mowy w toku.',
        'en': 'Speech recognition in progress.',
    },
    'server_disconnected': {
        'pl': 'Serwer rozłączony - przekroczono czas bezczynności.',
        'en': 'Server disconnected - maximum idle time exceeded.',
    },
    'pipe_error': {
        'pl': 'Błąd wewnętrzny przetwarzania potoku.',
        'en': 'Internal pipe processing error.',
    },
    'service_error': {
        'pl': 'Błąd usługi: ',
        'en': 'Service error: ',
    },
    'finalizing': {
        'pl': 'Kończenie transkrypcji...',
        'en': 'Finalizing...',
    },
    'file_ready': {
        'pl': 'Wczytano plik dźwiękowy. Gotowy do przetwarzania.',
        'en': 'Audio file opened. Ready to process.',
    },
    'transcript_file': {
        'pl': 'transkrypcja.txt',
        'en': 'transcript.txt',
    },
    'disclaimer': {
        'pl': 'Autorzy usługi nie ponoszą odpowiedzialności za generowane przez nią treści, '
        'ani za konsekwencje dalszego ich wykorzystania. '
        'Usługa nie przechowuje przesłanych nagrań ani rozpoznanych tekstów.',
        'en': 'The creators of the service bear no responsibility for the content it generates '
        'or for the consequences of its further use. '
        'The service does not store uploaded recordings or recognized texts.',
    },
    'tts_link': {
        'pl': 'Porównywarka syntezy mowy',
        'en': 'Speech synthesis comparison',
    },
    'rlhf_link': {
        'pl': 'Dostrajanie modelu ASR (RLHF)',
        'en': 'Fine-tuning the ASR model (RLHF)',
    },
    'invalid_audio': {
        'pl': 'Nieprawidłowy plik dźwiękowy.',
        'en': 'Invalid audio file.',
    },
    'tag': {
        'pl': '',
        'en': '',
    },
}

# Translation dictionary

STRINGS = {
    'pl': {key: val['pl'] for key, val in __STRING_TABLE.items()},
    'en': {key: val['en'] for key, val in __STRING_TABLE.items()},
}
