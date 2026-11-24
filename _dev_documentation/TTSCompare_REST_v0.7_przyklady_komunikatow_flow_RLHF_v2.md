# TTSCompare REST API v0.7 — przykłady komunikatów w przepływie RLHF ↔ TTS

**Stan:** 17 listopada 2026 r.  
**Środowisko:** laboratoryjna VM `ksmvm1`  
**Cel:** przykłady komunikatów HTTP i JSON wymienianych między klientem RLHF a usługą TTSCompare.

> **Jak czytać przykłady:** wszystkie identyfikatory `job_id`, daty, wielkości plików i sumy kontrolne w tym dokumencie są **ilustracyjne**, a nie skopiowane z rzeczywistego zadania. Przykłady zachowują strukturę kontraktu, lecz nie stanowią zrzutu odpowiedzi serwera bajt w bajt. Niektóre pola statusu mogą zależeć od etapu realizacji. Kod w blokach poniżej jest **treścią przykładowych komunikatów**, a nie skryptem do wykonania.

**Adres bazowy dostępny obecnie wyłącznie na VM:** `http://127.0.0.1:18774`. Gdy RLHF działa na innej maszynie lub w innym kontenerze, adres musi zostać zastąpiony uzgodnionym adresem dostępnym z jego środowiska. Dostęp zdalny nie jest jeszcze skonfigurowany.

## Konfiguracja silników i głosów

| `engine` | `voice_id` | WAV potwierdzone w testach |
|---|---|---|
| `omnivoice` | `Arek` | 24 000 Hz |
| `xtts` | `Arek` | 24 000 Hz |
| `chatterbox` | `Arek` | 24 000 Hz |
| `piper` | `pl_PL-gosia-medium` | 22 050 Hz |

Przykłady poniżej posługują się tym samym polskim tekstem dla czterech silników. W rzeczywistym wsadzie teksty mogą się różnić, a dany silnik może pojawić się wielokrotnie. **Aktualny limit: 100 pozycji na wsad; retencja: 86 400 sekund (24 h).**

---

## Punkt 1. Odczyt konfiguracji — opcjonalny

RLHF może pominąć ten krok, jeżeli ma aktualną, uzgodnioną konfigurację wszystkich par silnik–głos i zna limit wsadu.

**Żądanie:**

```http
GET /v1/capabilities HTTP/1.1
Host: 127.0.0.1:18774
```

**Przykładowa odpowiedź:**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "api_version": "1",
  "languages": ["pl"],
  "audio_formats": ["wav"],
  "archive_formats": ["zip"],
  "max_batch_size": 100,
  "result_retention_seconds": 86400,
  "engines": [
    {
      "id": "omnivoice",
      "mode": "real",
      "voices": [{ "id": "Arek" }]
    },
    {
      "id": "xtts",
      "mode": "real",
      "voices": [{ "id": "Arek" }]
    },
    {
      "id": "chatterbox",
      "mode": "real",
      "voices": [{ "id": "Arek" }]
    },
    {
      "id": "piper",
      "mode": "real",
      "voices": [{ "id": "pl_PL-gosia-medium" }]
    }
  ]
}
```

**Interpretacja:** do kolejnego kroku RLHF używa tylko dozwolonych par `engine` / `voice_id`. W realnej odpowiedzi mogą znajdować się dodatkowe pola informacyjne, np. wersje silników.

---

## Punkt 2. Wysłanie wsadu syntezy

**Żądanie:**

```http
POST /v1/jobs HTTP/1.1
Host: 127.0.0.1:18774
Content-Type: application/json
```

```json
{
  "request_id": "rlhf-batch-0001",
  "language": "pl",
  "audio_format": "wav",
  "archive_format": "zip",
  "items": [
    {
      "item_id": "item-001",
      "text": "To jest przykładowy tekst do syntezy mowy.",
      "engine": "omnivoice",
      "voice_id": "Arek"
    },
    {
      "item_id": "item-002",
      "text": "To jest przykładowy tekst do syntezy mowy.",
      "engine": "xtts",
      "voice_id": "Arek"
    },
    {
      "item_id": "item-003",
      "text": "To jest przykładowy tekst do syntezy mowy.",
      "engine": "chatterbox",
      "voice_id": "Arek"
    },
    {
      "item_id": "item-004",
      "text": "To jest przykładowy tekst do syntezy mowy.",
      "engine": "piper",
      "voice_id": "pl_PL-gosia-medium"
    }
  ]
}
```

**Przykładowa odpowiedź potwierdzająca przyjęcie:**

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
```

```json
{
  "job_id": "job-EXAMPLE-0001",
  "status": "queued"
}
```

**Interpretacja:** `202` oznacza przyjęcie zlecenia do asynchronicznego wykonania, a nie gotowe pliki. RLHF zapamiętuje `job_id` do sprawdzania statusu. `request_id` służy idempotencji: powtórzenie identycznego zlecenia nie powinno uruchamiać ponownej syntezy.

---

## Punkt 3. Sprawdzanie statusu zadania (polling)

**Żądanie:**

```http
GET /v1/jobs/job-EXAMPLE-0001 HTTP/1.1
Host: 127.0.0.1:18774
```

**Przykładowa odpowiedź podczas realizacji:**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "job_id": "job-EXAMPLE-0001",
  "status": "running",
  "total_items": 4,
  "processed_items": 2
}
```

**Przykładowa odpowiedź po pełnym sukcesie:**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "job_id": "job-EXAMPLE-0001",
  "status": "completed",
  "total_items": 4,
  "successful_items": 4,
  "failed_items": 0,
  "archive_url": "http://127.0.0.1:18774/v1/results/job-EXAMPLE-0001/archive",
  "archive_format": "zip",
  "byte_size": 950000,
  "sha256": "<rzeczywisty-64-znakowy-hash-SHA-256-archiwum>",
  "ready_at": "2026-11-17T08:30:00Z",
  "expires_at": "2026-11-18T08:30:00Z"
}
```

**Interpretacja:** RLHF ponawia odczyt, dopóki status jest `queued` albo `running`. Stany końcowe to `completed`, `completed_with_failures` i `failed`. `archive_url` z odpowiedzi wskazuje archiwum danego zadania. Jeżeli klient znajduje się poza VM, taki URL z `127.0.0.1` nie będzie dla niego osiągalny.

---

## Punkt 4. Pobranie archiwum wynikowego

**Żądanie:**

```http
GET /v1/results/job-EXAMPLE-0001/archive HTTP/1.1
Host: 127.0.0.1:18774
```

**Przykładowa odpowiedź:**

```http
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Length: 950000
```

Treścią odpowiedzi jest **binarny plik ZIP**, nie JSON. Przykładowa zawartość po rozpakowaniu:

```text
manifest.json
audio/file-000001.wav
audio/file-000002.wav
audio/file-000003.wav
audio/file-000004.wav
```

**Przykładowy `manifest.json` dla czterech udanych pozycji:**

```json
{
  "job_id": "job-EXAMPLE-0001",
  "items": [
    {
      "item_id": "item-001",
      "status": "success",
      "file": "audio/file-000001.wav",
      "engine": "omnivoice",
      "voice_id": "Arek",
      "real_tts": true,
      "sha256": "<hash-SHA-256-pierwszego-WAV>"
    },
    {
      "item_id": "item-002",
      "status": "success",
      "file": "audio/file-000002.wav",
      "engine": "xtts",
      "voice_id": "Arek",
      "real_tts": true,
      "sha256": "<hash-SHA-256-drugiego-WAV>"
    },
    {
      "item_id": "item-003",
      "status": "success",
      "file": "audio/file-000003.wav",
      "engine": "chatterbox",
      "voice_id": "Arek",
      "real_tts": true,
      "sha256": "<hash-SHA-256-trzeciego-WAV>"
    },
    {
      "item_id": "item-004",
      "status": "success",
      "file": "audio/file-000004.wav",
      "engine": "piper",
      "voice_id": "pl_PL-gosia-medium",
      "real_tts": true,
      "sha256": "<hash-SHA-256-czwartego-WAV>"
    }
  ]
}
```

**Interpretacja:** RLHF wiąże pliki z wejściowymi `item_id` na podstawie manifestu, a nie samej nazwy pliku lub kolejności wpisów. Powinien zachować nagrania we własnym systemie, ponieważ TTSCompare udostępnia je czasowo (24 godziny od `ready_at`). Pola dodatkowe manifestu (np. wersja silnika albo czasy syntezy) mogą występować zależnie od implementacji.

---

## Punkt 5. Obsługa częściowych błędów — wariant uzupełniający

Jeżeli np. pozycja `item-003` nie powiedzie się, ale trzy inne wygenerują nagrania, odpowiedź na **krok 3** może wyglądać następująco:

```json
{
  "job_id": "job-EXAMPLE-0002",
  "status": "completed_with_failures",
  "total_items": 4,
  "successful_items": 3,
  "failed_items": 1,
  "archive_url": "http://127.0.0.1:18774/v1/results/job-EXAMPLE-0002/archive"
}
```

Archiwum z **kroku 4** zawiera wtedy trzy pliki WAV i manifest dla **wszystkich czterech** pozycji. Fragment manifestu wskazujący błąd:

```json
{
  "item_id": "item-003",
  "status": "failed",
  "engine": "chatterbox",
  "voice_id": "Arek",
  "error": {
    "code": "synthesis_failed",
    "message": "Silnik nie mógł wygenerować nagrania dla tej pozycji.",
    "retryable": false
  }
}
```

Kod i opis błędu w tym fragmencie są **przykładowe**. Przy nieprawidłowym głosie w testach stosowano błąd pozycji `invalid_voice`.

### Co w przypadku 0 poprawnych plików?

W obecnej v0.7 status zadania to `failed`, nawet jeśli powstało archiwum zawierające wyłącznie manifest. **To znana różnica interpretacyjna** względem briefu RLHF, w którym `failed` zarezerwowano dla sytuacji bez użytecznego archiwum; wymaga uzgodnienia przed integracją.

---

## Punkt 6. Typowe sytuacje błędne i ponowienia

| Sytuacja | Oczekiwane zachowanie / potwierdzenie |
|---|---|
| Powtórzenie tego samego `request_id` i identycznych danych | Ponowna odpowiedź wskazująca istniejące `job_id`, bez nowej syntezy; potwierdzono w testach |
| Ten sam `request_id`, ale zmieniony wsad | `HTTP 409 Conflict`; potwierdzono w testach |
| Jedna pozycja z niepoprawnym `voice_id` | Błąd tej pozycji w manifeście, pozostałe mogą zakończyć się powodzeniem |
| Pobieranie ZIP po upływie `expires_at` | `HTTP 410 Gone`; potwierdzono w teście z wymuszonym wygaśnięciem |

**Przykładowe komunikaty statusowe bez zakładania dokładnego ciała błędu:**

```http
HTTP/1.1 409 Conflict
Content-Type: application/json
```

```http
HTTP/1.1 410 Gone
Content-Type: application/json
```

Dokładne wartości pól `error.code` i `error.message` należy odczytywać z rzeczywistej odpowiedzi. Nie należy opierać logiki klienta na wymyślonych kodach błędów z przykładów.

---

## Skrót sekwencji

1. **[Opcjonalnie]** `GET /v1/capabilities` → silniki, głosy, limity.
2. **`POST /v1/jobs`** → `HTTP 202`, `job_id`.
3. **`GET /v1/jobs/{job_id}`** → `queued` / `running` → `completed` albo status błędu.
4. **`GET /v1/results/{job_id}/archive`** → ZIP (WAV + `manifest.json`).
5. **Po stronie RLHF:** weryfikacja archiwum, mapowanie plików po `item_id` i trwałe zapisanie wyników.

**Zakres gotowości:** główny przepływ został przetestowany lokalnie na VM `ksmvm1` dla czterech rzeczywistych silników. Przykładowe komunikaty nie oznaczają, że dostęp sieciowy dla zdalnego klienta RLHF został już skonfigurowany lub przetestowany.
