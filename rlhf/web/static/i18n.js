"use strict";
/* =========================================================
   CAISE-ASR — i18n / language-switching module
   Supports Polish (pl) and English (en).
   Exposes:  window.setLang(lang)
             window.t(key)  — translation lookup
   ========================================================= */
(function () {

  const T = {
    pl: {
      /* stats bar (annotate page) */
      'stats.pending':      'Oczekujące',
      'stats.annotated':    'Ocenione',
      'stats.skipped':      'Pominięte',
      'stats.audio':        'Pliki audio',
      /* ref toggle */
      'ref.reveal':         '(kliknij, aby pokazać)',
      'ref.hide':           '(kliknij, aby ukryć)',
      'ref.none':           '(brak transkrypcji referencyjnej)',
      'error.audio':        '[!] Nie udało się wczytać nagrania',
      'error.loadPair':     'Nie udało się wczytać pary',
      'error.submit':       'Nie udało się zapisać adnotacji',
      /* exclude dialog */
      'exclude.prompt':     'Powód wykluczenia nagrania (np. szum, niewłaściwy język, uszkodzone nagranie):',
      'exclude.default':    'szum',
      'corpora.activateConfirm': 'Włączyć ten zbiór do przyszłego wyboru danych? Wykluczenia nadal obowiązują.',
      'corpora.deactivateConfirm': 'Wyłączyć ten zbiór z przyszłego wyboru danych? Bieżący trening nie zostanie przerwany.',
    },
    en: {
      'stats.pending':      'Pending',
      'stats.annotated':    'Annotated',
      'stats.skipped':      'Skipped',
      'stats.audio':        'Audio files',
      'ref.reveal':         '(click to reveal)',
      'ref.hide':           '(click to hide)',
      'ref.none':           '(no reference transcript)',
      'error.audio':        '[!] Audio failed to load',
      'error.loadPair':     'Error loading pair',
      'error.submit':       'Submit failed',
      'exclude.prompt':     'Reason for excluding this recording (e.g. noise, wrong language, corrupt):',
      'exclude.default':    'noise',
      'corpora.activateConfirm': 'Enable this corpus for future data selection? Existing exclusions still apply.',
      'corpora.deactivateConfirm': 'Disable this corpus for future data selection? Running training will not be interrupted.',
    }
  };

  function getLang() {
    return document.documentElement.getAttribute('data-lang') || 'pl';
  }

  function applyLang(lang) {
    if (lang !== 'pl' && lang !== 'en') lang = 'pl';
    document.documentElement.setAttribute('data-lang', lang);
    try { localStorage.setItem('caise_lang', lang); } catch (_) { /* private browsing */ }

    var plLink = document.getElementById('lang-pl-link');
    var enLink = document.getElementById('lang-en-link');
    if (plLink) plLink.classList.toggle('lang-active', lang === 'pl');
    if (enLink) enLink.classList.toggle('lang-active', lang === 'en');
  }

  /* Public API */
  window.setLang = function (lang) { applyLang(lang); };

  window.t = function (key) {
    var lang = getLang();
    var dict = T[lang] || T['pl'];
    return (dict[key] != null) ? dict[key] : ((T['en'][key] != null) ? T['en'][key] : key);
  };

  /* Init — restore saved preference, fall back to Polish */
  var saved = null;
  try { saved = localStorage.getItem('caise_lang'); } catch (_) {}
  var init = (saved === 'pl' || saved === 'en') ? saved : 'pl';

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { applyLang(init); });
  } else {
    applyLang(init);
  }

})();
