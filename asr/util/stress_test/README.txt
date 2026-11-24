Instrukcja testu obciążeniowego usługi CAISE-ASR
================================================

Autor: Grzegorz Szwoch, grzszwoc@pg.edu.pl
Katedra Systemów Multimedialnych PG
23.09.2026


Przygotowanie obrazu Dockera
----------------------------

Oprogramowanie zostało przekazane w pliku asr_workflow.tar.gz.
Aktualna wersja oprogramowania jest dostępna w repozytorium:
https://projects.task.gda.pl/caise/spaces/pl0161-01/asr_workflow

Do poprawnego działania potrzebne są:
- adres IP, do którego będzie możliwy dostęp z komputera uruchamiającego test,
- odblokowany dostęp do portu 8000, przynajmniej dla komputera testowego.

Zbudowanie obrazu Dockera - w katalogu asr_workflow:

docker build -t asr_workflow .

Uruchomienie obrazu:

docker run -d --restart unless-stopped --gpus all -p 8000:8000 --shm-size=2gb --name asr_workflow asr_workflow

Jest przygotowany skrypt asrdock, który ułatwia wykonywanie operacji.
./asrdock build - zbudowanie obrazu.
./asrdock daemon - uruchomienie obrazu.
Inne operacje - opisane w skrypcie.


Uruchomienie procedury testowej
-------------------------------

Procedurę testową należy uruchamiać z poziomu komputera, z kórego możliwy jest dostęp 
do maszyny z uruchomionym kontenerem Dockera aplikacji.

Wymagania: Python z zainstalowanym modułem requests.

Skrypt testowy znajduje się w pliku stress_test_asr.py. 
Konfiguracja jest zawarta w początkowej części skryptu.

URL - adres URL do maszyny z uruchomioną aplikacją (z numerem portu).

NUM_CLIENTS - liczba klientów, które będą równolegle wysyłać żądania do aplikacji.
              Zaleca się aby NUM_CLIENTS było co najmniej równe liczbie GPU
              dostępnych dla aplikacji.

TEST_DURATION - czas trwania testu w sekundach.

WAVDIR - katalog zawierający testowe nagrania mowy w formacie WAV lub MP3.

Wraz ze skryptem dostarczany jest katalog test-dataset, zawierający 100 plików
z nagraniami mowy polskiej, medycznej.

Możliwe jest wykorzystanie własnych plików. Nie zaleca się używania dużych plików,
ponieważ ich przesyłanie do aplikacji będzie trwało długo, a w tym czasie aplikacja
nie będzie obciążała GPU.

Uruchomienie skryptu stress_test_asr.py rozpoczyna procedurę testową.
