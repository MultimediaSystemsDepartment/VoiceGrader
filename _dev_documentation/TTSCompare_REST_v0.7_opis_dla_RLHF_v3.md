# TTSCompare – REST API v0.7 (REAL_ONLY)

**Krótki opis korzystania z wersji testowej**  
**Stan na:** 17 listopada 2026 r.  
**Środowisko:** VM `ksmvm1`, Docker

## 1. Cel

Usługa TTSCompare przyjmuje od systemu RLHF wsady polskich tekstów wraz z wyborem silnika i głosu. Wykonuje syntezę asynchronicznie, udostępnia stan realizacji, a po zakończeniu przygotowuje jedno archiwum ZIP z nagraniami WAV i manifestem. Wersja v0.7 korzysta wyłącznie z rzeczywistych silników TTS, bez nagrań zastępczych (mock).

**Status:** podstawowe funkcje API zostały przetestowane na VM i są gotowe do testów integracyjnych. Nie oznacza to jeszcze uruchomionej integracji z działającym systemem RLHF.

## 2. Dostępne silniki

| Silnik | Dostępny głos | Częstotliwość WAV potwierdzona w testach |
|---|---|---|
| OmniVoice | Arek | 24 kHz |
| XTTS | Arek | 24 kHz |
| Chatterbox | Arek | 24 kHz |
| Piper | pl_PL-gosia-medium | 22,05 kHz |

**Konfiguracja wspólna dla obu wariantów integracji:** głos **Arek** jest dostępny w trzech silnikach: **OmniVoice, XTTS i Chatterbox**. Piper korzysta z głosu **pl_PL-gosia-medium**. Identyfikatory `engine` i `voice_id` należy podawać dokładnie w powyższej postaci; głosy są przypisane do konkretnych silników.

RLHF może korzystać z **konfiguracji statycznej** (bezpośrednio używa uzgodnionych par silnik–głos) lub **dynamicznej** (przed wysłaniem zadania pobiera aktualne pary i limit z `GET /v1/capabilities`). Odczyt listy nie jest obowiązkowy przy każdym wsadzie, jeśli konfiguracja jest znana i aktualna.

## 3. Podział odpowiedzialności

**System RLHF:** wybiera teksty i głosy, dzieli kampanię na wsady, wysyła zadania, sprawdza ich stan, pobiera archiwum, dopasowuje pliki do pozycji wejściowych według manifestu i przechowuje nagrania po swojej stronie.

**Usługa TTSCompare:** udostępnia listę silników i głosów, przyjmuje zadania wsadowe, realizuje syntezę, raportuje postęp, przygotowuje archiwum wynikowe oraz zapewnia czasowy dostęp do archiwum.

RLHF nie przesyła nagrań referencyjnych ani próbek do klonowania głosów. Głosy przygotowywane są po stronie TTSCompare.

## 4. Dostępne operacje REST API

| Operacja | Przeznaczenie |
|---|---|
| GET /v1/capabilities | Informacje o silnikach, głosach, formatach i limitach |
| POST /v1/jobs | Zlecenie asynchronicznej syntezy wsadu |
| GET /v1/jobs/{job_id} | Odczyt statusu i informacji o wyniku zadania |
| GET /v1/results/{job_id}/archive | Pobranie archiwum ZIP dla zakończonego zadania |

Dodatkowo dostępne jest GET /health, służące do podstawowego sprawdzenia działania API. Poprawna odpowiedź tego punktu nie gwarantuje chwilowej dostępności wszystkich silników syntezy.

## 5. Sposób korzystania

1. Opcjonalnie RLHF pobiera listę silników i głosów (przy znanej konfiguracji statycznej pomija ten krok).
2. RLHF wysyła wsad z identyfikatorem żądania oraz pozycjami zawierającymi identyfikator tekstu, tekst, silnik i głos.
3. TTSCompare zwraca identyfikator zadania. Synteza odbywa się asynchronicznie.
4. RLHF cyklicznie sprawdza stan zadania do czasu jego zakończenia.
5. RLHF pobiera pojedyncze archiwum ZIP i odczytuje manifest.
6. RLHF przyporządkowuje nagrania WAV do przesłanych pozycji i zapisuje je na stałe po swojej stronie.

### Specyfikacja kroku 2 – wysłanie wsadu

RLHF wywołuje **POST /v1/jobs** z treścią w formacie **JSON (UTF-8)**. Zgłoszenie zawiera pola:

| Pole | Opis | Wartość lub zasada |
|---|---|---|
| request_id | Identyfikator całego wsadu nadawany przez RLHF | Nowy dla nowego wsadu; ten sam przy ponowieniu identycznego żądania |
| language | Język syntezy | pl |
| audio_format | Format wynikowych nagrań | wav |
| archive_format | Format zbiorczego wyniku | zip |
| items | Lista pozycji do syntezy | Maksymalnie 100 w aktualnej wersji |
| item_id | Identyfikator konkretnej pozycji na liście | Unikalny w obrębie wsadu, zachowany w manifeście |
| text | Tekst danej pozycji | Dowolny tekst przeznaczony do syntezy w języku polskim |
| engine | Silnik dla danej pozycji | omnivoice, xtts, chatterbox albo piper |
| voice_id | Głos dla wskazanego silnika | Arek dla trzech pierwszych silników; pl_PL-gosia-medium dla Piper |

Przykładowy wsad czteroelementowy zawiera cztery pozycje z identycznym polskim zdaniem, po jednej na silnik: **item-001 → omnivoice / Arek**, **item-002 → xtts / Arek**, **item-003 → chatterbox / Arek**, **item-004 → piper / pl_PL-gosia-medium**. Każda pozycja może też zawierać inny tekst, a jeden silnik może występować wielokrotnie.

Po przyjęciu zadania TTSCompare zwraca **HTTP 202 Accepted**, `job_id` i status początkowy (np. `queued`). Synteza przebiega asynchronicznie. RLHF następnie odczytuje stan zadania przez **GET /v1/jobs/{job_id}**.

Identyfikator żądania służy do bezpiecznego ponawiania tego samego zlecenia bez ponownego uruchamiania syntezy. Ponowienie z innymi danymi, lecz z tym samym `request_id`, skutkowało w testach odpowiedzią HTTP 409. Każda pozycja ma własny identyfikator, zwracany w wynikach.

## 6. Wyniki, statusy i błędy

Możliwe stany zadania to: **queued**, **running**, **completed**, **completed_with_failures** i **failed**.

Archiwum ZIP zawiera manifest z przypisaniem plików WAV do identyfikatorów pozycji, informacją o użytym silniku i głosie oraz opisami błędów dla pozycji nieudanych. Błąd pojedynczej pozycji nie musi blokować poprawnych wyników pozostałych pozycji wsadu.

Wersja testowa obsługuje język polski, format WAV oraz archiwum ZIP. Aktualnie deklarowany maksymalny wsad to **100 pozycji**. Okres dostępności archiwum wynosi **24 godziny od jego przygotowania**; po upływie tego czasu API odmawia pobrania (HTTP 410).

## 7. Stan testów i ograniczenia

**Potwierdzone na VM:** rzeczywista synteza wszystkimi czterema silnikami, obsługa wsadu z pozycją błędną, idempotencja żądań, pobieranie ZIP, weryfikacja manifestu i sum kontrolnych oraz zwracanie HTTP 410 po wygaśnięciu dostępu. Osobno przetestowano działanie n8n na tych samych czterech silnikach.

**Przed testami zdalnymi:** API jest obecnie dostępne tylko lokalnie na VM `ksmvm1`, pod adresem `http://127.0.0.1:18774`. Trzeba uzgodnić osiągalny dla klienta RLHF adres i bezpieczny sposób połączenia. Usługa w tej wersji nie wymaga uwierzytelniania, więc nie powinna być wystawiana do niezaufanej sieci.

**Znane różnice i kwestie do uzgodnienia:**

- W przypadku braku jakiegokolwiek poprawnego WAV wersja v0.7 zwraca status **failed**, nawet jeżeli przygotowano ZIP z samym manifestem. Dokument RLHF rezerwuje ten status dla sytuacji bez użytecznego archiwum; interpretację należy uzgodnić.
- Po wygaśnięciu archiwum dostęp HTTP jest blokowany, ale automatyczne fizyczne usuwanie plików nie zostało jeszcze wdrożone.
- Nie przeprowadzono jeszcze testu z rzeczywistym klientem RLHF uruchomionym na innym hoście ani pełnego testu dużych kampanii wsadowych.

## 8. Zakres gotowości

**Wersja TTSCompare REST API v0.7 jest gotowa do rozpoczęcia testów integracyjnych według głównych założeń dokumentu RLHF.** Aktualny zakres to środowisko testowe VM. Uruchomienie dostępu zewnętrznego i potwierdzenie pełnej zgodności integracyjnej stanowią osobny etap.
