"""
wer.py
Compute WER and CER for the recognized text.
Author: Grzegorz Szwoch, KSM PG, 2026.
"""

import jiwer
from whisper_normalizer.basic import BasicTextNormalizer


def normalize_text(txt):
    """Perform text normalization for WER/CER calculation."""
    normalizer = BasicTextNormalizer()
    output = normalizer(txt)
    return output


def compute_metrics(recognized, reference, normalize=True):
    """Compute WER and CER for the recognized text,
    compared with the reference text.
    """
    if normalize:
        recognized = normalize_text(recognized)
        reference = normalize_text(reference)
    wer_score = jiwer.wer(reference, recognized)
    cer_score = jiwer.cer(reference, recognized)
    res = {'wer': wer_score, 'cer': cer_score, 'recognized': recognized, 'reference': reference}
    return res


if __name__ == '__main__':
    from pathlib import Path

    def test_wer(rec, ref):
        recognized = Path(rec).read_text()
        reference = Path(ref).read_text()
        res = compute_metrics(recognized, reference)
        print('Recognized:', res['recognized'], sep='\n')
        print('\nReference:', res['reference'], sep='\n')
        print(f'\nWER = {100 * res["wer"]:.2f}, CER = {100 * res["cer"]:.2f}')

    # test_wer('../../data/wer/wh_rec.txt', '../../data/wer/wh_ref.txt')
    test_wer('../../data/wer/whisperx.txt', '../../data/wer/wh_ref.txt')
